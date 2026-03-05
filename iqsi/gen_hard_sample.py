import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from PIL import Image
import torchvision.transforms.functional as TF
import random
from torch.transform import v2

REALISTIC_STYLE_POOL = [
    "sunlight, hard shadows, high contrast, outdoor photography",
    "overcast sky, soft lighting, diffused light, flat lighting",
    "golden hour, warm tone, morning light",
    "wet surface, rain droplets, damp, glossy finish",
    "dry texture, rough surface, gritty, detailed",
    "dirty, weathered, worn out, dusty",
    "macro photography, extreme close-up, sharp texture focus",
    "wide angle lens, top down view, flat lay",
    "flash photography, harsh lighting, direct light"
]

@torch.no_grad()
def gen_hard_samples(
    model, 
    pipe, 
    synth_images_01,
    synth_labels, 
    real_images, 
    real_labels,
    classes,
    config, 
    accelerator, 
    opt_steps,
    centroids_tensor
):
    device = accelerator.device
    target_size = (512, 512)
    vae_dtype = pipe.vae.dtype
    unet_dtype = pipe.unet.dtype
    alpha = 0.01

    synth_images_vae = F.interpolate(synth_images_01, size=target_size, mode='bilinear', align_corners=False)
    synth_images_vae = synth_images_vae * 2.0 - 1.0 
    synth_images_vae = synth_images_vae.to(device=device, dtype=vae_dtype)

    with torch.no_grad():
        latents = pipe.vae.encode(synth_images_vae).latent_dist.sample() * pipe.vae.config.scaling_factor
        latents = latents.to(dtype=unet_dtype)

    synth_classes = [classes[i.item()] for i in synth_labels]
    selected_style = random.choice(REALISTIC_STYLE_POOL)
    prompts = [f"texture of {c}, {selected_style}, organic, natural texture, raw material, highly detailed, 4k" for c in synth_classes]
    neg_prompts = ["ugly, tiling, poorly drawn, out of frame, disfigured, deformed, blur, watermark, grainy, painting, drawing, illustration"] * len(synth_classes)

    with torch.no_grad():
        text_inputs = pipe.tokenizer(prompts, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt").to(device)
        text_embeddings_cond = pipe.text_encoder(text_inputs.input_ids)[0]
        uncond_inputs = pipe.tokenizer(neg_prompts, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt").to(device)
        uncond_embeddings = pipe.text_encoder(uncond_inputs.input_ids)[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings_cond], dim=0).to(dtype=unet_dtype)

    noise_strength = 0.3
    num_inference_steps = 50
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    
    init_timestep = int(num_inference_steps * noise_strength)
    t_start = pipe.scheduler.timesteps[num_inference_steps - init_timestep]
    
    noise = torch.randn_like(latents)
    latents_t = pipe.scheduler.add_noise(latents, noise, torch.tensor([t_start], device=device))

    steps_to_run = pipe.scheduler.timesteps[num_inference_steps - init_timestep:]
    for t in steps_to_run:
        with torch.no_grad():
            latent_model_input = torch.cat([latents_t] * 2)
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
            latents_t = pipe.scheduler.step(noise_pred, t, latents_t).prev_sample

    final_images = None
    
    for k in range(opt_steps):
        latents_t = latents_t.detach().requires_grad_(True)
        
        with torch.autocast("cuda", dtype=torch.float16):
            decoded = pipe.vae.decode(latents_t / pipe.vae.config.scaling_factor).sample
        
        images = (decoded / 2 + 0.5).clamp(0, 1)
        images_clip = F.interpolate(images, size=(224, 224), mode='bilinear')
        images_trans = v2.functional.normalize(images_clip, mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

        out = model(images_trans, output_features=True)
        synth_logits, synth_feats = out["logits"], out["image_feats"]
        
        with torch.no_grad():
            real_out = model(real_images, output_features=True)
            real_logits, real_feats = real_out["logits"], real_out["image_feats"]
            L_real_vec = F.cross_entropy(real_logits, real_labels, reduction="none")

        L_synth_vec = F.cross_entropy(synth_logits, synth_labels, reduction="none")
        synth_loss = L_synth_vec.mean()

        eps_GS = torch.tensor(0.0, device=device)
        g = 0
        
        dists_real = torch.cdist(real_feats.float(), centroids_tensor.float())
        real_regions = torch.argmin(dists_real, dim=1)
        dists_synth = torch.cdist(synth_feats.float(), centroids_tensor.float())
        synth_regions = torch.argmin(dists_synth, dim=1)

        for i in range(config.train.num_clusters):
            r_idx = (real_regions == i).nonzero(as_tuple=True)[0]
            s_idx = (synth_regions == i).nonzero(as_tuple=True)[0]
            num_s = len(s_idx)
            
            if num_s > 0 and len(r_idx) > 0:
                diff = torch.abs(L_real_vec[r_idx].unsqueeze(1) - L_synth_vec[s_idx].unsqueeze(0))
                eps_GS += diff.mean() * num_s
                g += num_s

        if g > 0:
            eps_GS /= g

        loss = -config.train.lamda4 * synth_loss + config.train.lamda5 * eps_GS

        loss.backward()
        with torch.no_grad():
            latents_t -= alpha * latents_t.grad
            latents_t.grad.zero_()

        if k == opt_steps - 1:
            final_images = images.detach().clone()

    return final_images