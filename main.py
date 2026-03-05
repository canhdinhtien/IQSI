import gc
import os
import datetime
import tqdm
import torch
import contextlib
import wandb

from collections import defaultdict
from functools import partial

from ml_collections import config_flags
from flask import config
from absl import app, flags

from accelerate import Accelerator
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger

from data import get_data_loader, get_synth_train_data_loader

from diffusers import StableDiffusionPipeline, DDIMScheduler
from models import TinyDecoder, CLIP, evaluate
from cluster import get_centroids_from_loader
from data import get_transforms

import torch.nn.functional as F

from iqsi import train_step, train_step_with_hard_samples
tqdm = partial(tqdm.tqdm, dynamic_ncols=True)

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/config.py", "Training configuration.")

logger = get_logger(__name__)

def main(argv):
    del argv
    config = FLAGS.config
    
    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    if config.resume_from:
        config.resume_from = os.path.normpath(os.path.expanduser(config.resume_from))
        if "checkpoint_" not in os.path.basename(config.resume_from):
            # get the most recent checkpoint in this directory
            checkpoints = list(
                filter(lambda x: "checkpoint_" in x, os.listdir(config.resume_from))
            )
            if len(checkpoints) == 0:
                raise ValueError(f"No checkpoints found in {config.resume_from}")
            config.resume_from = os.path.join(
                config.resume_from,
                sorted(checkpoints, key=lambda x: int(x.split("_")[-1]))[-1],
            )

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config
    )

    set_seed(config.seed, device_specific=True)

    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="IQSI",
            config=config.to_dict(),
            init_kwargs={"wandb": {"name": config.run_name}},
        )
    logger.info(f"\n{config}")

    real_train_loader, test_loader = get_data_loader(
        real_train_data_dir=config.path.real_train_dir,
        metadata_dir=config.path.metadata_dir,
        dataset=config.dataset_name,
        bs=config.train.real_batch_size,
        n_img_per_cls=config.n_shot,
        model_type=config.model_type,
        is_rand_aug=config.is_random_aug
    )

    classes = real_train_loader.dataset.classes
    real_train_loader, test_loader = accelerator.prepare(real_train_loader, test_loader)

    synth_train_loader = get_synth_train_data_loader(
        synth_train_data_dir=config.path.synthesis_dir,
        bs=config.train.synth_batch_size,
        is_rand_aug=config.train.is_rand_aug,
        target_label=config.train.target_label,
        n_img_per_cls=config.train.n_img_per_cls,
        dataset=config.dataset_name,
        n_shot=config.n_shot,
        real_train_fewshot_data_dir=config.path.fewshot_dir,
        is_pooled_fewshot=config.is_pooled_fewshot,
        model_type=config.model_type
    )

    synth_train_loader = accelerator.prepare(synth_train_loader)

    def infinite_iterable(loader):
        while True:
            for batch in loader:
                yield batch

    synth_iter = iter(infinite_iterable(synth_train_loader))

    pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained.model, revision=config.pretrained.revision
    )

    pipeline.load_lora_weights(".", weight_name=config.unet.weight_lora)
    pipeline.fuse_lora(lora_scale=1.0)
    pipeline.unload_lora_weights()

    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(False)

    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)

    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    pipeline.vae.to(accelerator.device, dtype=inference_dtype)
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)
    if config.use_lora:
        pipeline.unet.to(accelerator.device, dtype=inference_dtype)

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    tiny_decoder = None
    if config.pretrained.use_tiny_decoder == True:
        tiny_decoder = TinyDecoder(in_ch=4, base_ch=128)

    model = CLIP(
        dataset=config.dataset_name,
        is_lora_image=config.classifier.is_lora_image,
        is_lora_text=config.classifier.is_lora_text,
        clip_download_dir=config.classifier.clip_download_dir,
        clip_version=config.classifier.clip_version,
        precomputed_text_embs_path=config.classifier.precomputed_text_embs_path
    )

    model = accelerator.prepare(model)

    dtype_clip = model.clip.visual.conv1.weight.dtype

    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        model.learnable_params(),
        lr=config.train.lr,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    autocast = contextlib.nullcontext() if config.use_lora else accelerator.autocast()

    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        accelerator.load_state(config.resume_from)
        first_epoch = int(config.resume_from.split("_")[-1]) + 1
    else:
        first_epoch = 0

    step = 0

    all_real_paths = []
    for batch in real_train_loader:
        _, _, paths = batch 
        all_real_paths.extend(paths)

    _, clean_transform = get_transforms(config.model_type)
    
    centroids = get_centroids_from_loader(model, all_real_paths, config.train.num_clusters, clean_transform, config.train.real_batch_size, accelerator, dtype_clip)
    centroids_tensor = torch.from_numpy(centroids).to(accelerator.device)

    for epoch in range(first_epoch, config.train.num_epochs):
        if epoch < config.train.num_epochs_warm_up:
            for real_batch in real_train_loader:
                step += 1
                
                synth_batch = next(synth_iter)
                
                train_logs = train_step(
                    model, real_batch, synth_batch, centroids_tensor, 
                    optimizer, accelerator, config, dtype_clip, autocast
                )
                
                if accelerator.is_main_process:
                    accelerator.log(train_logs, step=step)
                    
                if step % config.train.gc_steps == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
        elif epoch % 2 == 0:
            for real_batch in real_train_loader:
                step += 1
                synth_batch = next(synth_iter)

                train_logs = train_step_with_hard_samples(
                    model, pipeline, real_batch, synth_batch, centroids_tensor, 
                    optimizer, accelerator, config, dtype_clip, autocast, config.train.prop_hard, classes
                )
                
                if accelerator.is_main_process:
                    accelerator.log(train_logs, step=step)

                if step % config.gc_steps == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
        else:
            for real_batch in real_train_loader:
                step += 1
                
                synth_batch = next(synth_iter)
                
                train_logs = train_step(
                    model, real_batch, synth_batch, centroids_tensor, 
                    optimizer, accelerator, config, dtype_clip, autocast
                )
                
                if accelerator.is_main_process:
                    accelerator.log(train_logs, step=step)
                
                if step % config.train.gc_steps == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
        
        test_acc = evaluate(model, test_loader, accelerator)
        train_acc = evaluate(model, real_train_loader, accelerator)

        accelerator.log({
            "eval/test_accuracy": test_acc,
            "eval/train accuracy": train_acc
        }, step=epoch)

        if (epoch + 1) % config.train.update_centroids_freq == 0:
            centroids = get_centroids_from_loader(model, all_real_paths, config.train.num_clusters, clean_transform, config.train.real_batch_size, accelerator, dtype_clip)
            centroids_tensor = torch.from_numpy(centroids).to(accelerator.device)

if __name__ == "__main__":
    app.run(main)