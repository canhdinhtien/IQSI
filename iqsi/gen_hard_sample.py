import torch
import torch.nn.functional as F
from torchvision.transforms import v2
import gc

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
    centroids_tensor
):
    device = accelerator.device
    target_size = (224, 224)
    vae_dtype = pipe.vae.dtype 
    unet_dtype = pipe.unet.dtype
    alpha = 0.01
    generator = torch.Generator(device=device).manual_seed(config.seed)

    with torch.no_grad():
        synth_images_vae = F.interpolate(synth_images_01, size=target_size, mode='bilinear')
        synth_images_vae = (synth_images_vae * 2.0 - 1.0).to(dtype=vae_dtype)

        latents = pipe.vae.encode(synth_images_vae).latent_dist.sample() * pipe.vae.config.scaling_factor
        latents = latents.to(dtype=unet_dtype)

        synth_classes = [classes[i.item()] for i in synth_labels]
        style_idx = torch.randint(0, len(REALISTIC_STYLE_POOL), (1,), generator=generator).item()
        selected_style = REALISTIC_STYLE_POOL[style_idx]
        prompts = [f"texture of {c}, {selected_style}, organic, natural texture" for c in synth_classes]
        neg_prompts = ["ugly, blur, painting, drawing"] * len(synth_classes)

        text_inputs = pipe.tokenizer(prompts, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt").to(device)
        text_embeddings_cond = pipe.text_encoder(text_inputs.input_ids)[0]
        uncond_inputs = pipe.tokenizer(neg_prompts, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True, return_tensors="pt").to(device)
        uncond_embeddings = pipe.text_encoder(uncond_inputs.input_ids)[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings_cond], dim=0)

        noise_strength = 0.3
        num_inference_steps = 50
        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        init_timestep = int(num_inference_steps * noise_strength)
        t_start = pipe.scheduler.timesteps[num_inference_steps - init_timestep]
        
        noise = torch.randn(latents.shape, generator=generator, device=device, dtype=latents.dtype)
        latents_t = pipe.scheduler.add_noise(latents, noise, torch.tensor([t_start], device=device))

        steps_to_run = pipe.scheduler.timesteps[num_inference_steps - init_timestep:]
        for t in steps_to_run:
            latent_model_input = torch.cat([latents_t] * 2)
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
            latents_t = pipe.scheduler.step(noise_pred, t, latents_t).prev_sample
            
            latents_t = latents_t.detach()

    del text_embeddings, noise_pred, text_embeddings_cond, uncond_embeddings
    gc.collect()
    torch.cuda.empty_cache()

    final_images = None

    with torch.no_grad():
        real_out = model(real_images, output_features=True)
        L_real_vec = F.cross_entropy(real_out["logits"], real_labels, reduction="none")
        real_feats = real_out["image_feats"]
        real_feats = F.normalize(real_feats.float(), p=2, dim=1)

        centroids_tensor = F.normalize(centroids_tensor.float(), p=2, dim=1)

    dists_real = torch.cdist(real_feats.float(), centroids_tensor.float())
    real_regions = torch.argmin(dists_real, dim=1)

    for k in range(config.train.opt_steps):
        latents_t = latents_t.detach().clone().requires_grad_(True)
        
        with torch.autocast("cuda", dtype=torch.float16):
            decoded = pipe.vae.decode(latents_t / pipe.vae.config.scaling_factor).sample
        
        images = (decoded / 2 + 0.5).clamp(0, 1)
        images_clip = F.interpolate(images, size=(224, 224), mode='bilinear')
        images_trans = v2.functional.normalize(images_clip, mean=[0.481, 0.457, 0.408], std=[0.268, 0.261, 0.275])

        out = model(images_trans, output_features=True)
        synth_logits, synth_feats = out["logits"], out["image_feats"]
        
        synth_feats = F.normalize(synth_feats.float(), p=2, dim=1)

        L_synth_vec = F.cross_entropy(synth_logits, synth_labels, reduction="none")
        synth_loss = L_synth_vec.mean()

        eps_GS = torch.tensor(0.0, device=device)
        g = 0
        
        dists_synth = torch.cdist(synth_feats.float(), centroids_tensor.float())
        synth_regions = torch.argmin(dists_synth, dim=1)

        for i in range(config.train.num_clusters):
            r_idx = (real_regions == i).nonzero(as_tuple=True)[0]
            s_idx = (synth_regions == i).nonzero(as_tuple=True)[0]
            if len(s_idx) > 0 and len(r_idx) > 0:
                diff = torch.abs(L_real_vec[r_idx].unsqueeze(1) - L_synth_vec[s_idx].unsqueeze(0))
                eps_GS += diff.mean() * len(s_idx)
                g += len(s_idx)

        if g > 0:
            eps_GS /= g

        loss = -config.train.lamda4 * synth_loss + config.train.lamda5 * eps_GS
        loss.backward()
        
        with torch.no_grad():
            latents_t = (latents_t - alpha * latents_t.grad).detach()
            
        if k == config.train.opt_steps - 1:
            final_images = images.detach().clone()

    return final_images