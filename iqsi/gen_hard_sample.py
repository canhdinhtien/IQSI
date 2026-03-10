import torch
import torch.nn.functional as F
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

@torch.no_grad()
def precompute_assets(pipe, synth_images_01, synth_labels, classes, selected_style, device):
    target_size = (512, 512)
    synth_images_vae = F.interpolate(synth_images_01, size=target_size, mode='bilinear', align_corners=False)
    synth_images_vae = (synth_images_vae * 2.0 - 1.0).to(dtype=pipe.vae.dtype)
    
    latents = pipe.vae.encode(synth_images_vae).latent_dist.sample() * pipe.vae.config.scaling_factor
    
    synth_classes = [classes[i.item()] for i in synth_labels]
    prompts = [f"texture of {c}, {selected_style}, organic, natural texture" for c in synth_classes]
    neg_prompts = ["ugly, blur, painting, drawing"] * len(synth_classes)

    def encode_text(p_list):
        inputs = pipe.tokenizer(p_list, padding="max_length", max_length=pipe.tokenizer.model_max_length, 
                                truncation=True, return_tensors="pt").to(device)
        return pipe.text_encoder(inputs.input_ids)[0]

    cond_embeds = encode_text(prompts)
    uncond_embeds = encode_text(neg_prompts)
    text_embeddings = torch.cat([uncond_embeds, cond_embeds], dim=0)
    
    return latents.to(dtype=pipe.unet.dtype), text_embeddings

@torch.no_grad()
def diffusion_generate(pipe, latents, text_embeddings, noise_strength, generator, device):
    num_inference_steps = 50
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    init_step = int(num_inference_steps * noise_strength)
    t_start = pipe.scheduler.timesteps[-init_step]

    noise = torch.randn_like(latents, generator=generator)
    latents_t = pipe.scheduler.add_noise(latents, noise, torch.tensor([t_start], device=device))

    for t in pipe.scheduler.timesteps[-init_step:]:
        latent_model_input = torch.cat([latents_t] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
        
        latents_t = pipe.scheduler.step(noise_pred, t, latents_t).prev_sample
        
    return latents_t

def gen_hard_samples(model, pipe, synth_images_01, synth_labels, real_images, real_labels, 
                     classes, config, accelerator, centroids_tensor):
    device = accelerator.device
    alpha = 0.01 
    generator = torch.Generator(device=device).manual_seed(config.seed)
    num_samples = synth_images_01.shape[0]

    if torch.cuda.is_available():
        pipe.unet.enable_xformers_memory_efficient_attention()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

    idx = torch.randint(0, len(REALISTIC_STYLE_POOL), (1,)).item()
    selected_style = REALISTIC_STYLE_POOL[idx]
    with torch.no_grad():
        latents, text_embeddings = precompute_assets(pipe, synth_images_01, synth_labels, classes, selected_style, device)
        latents_t = diffusion_generate(pipe, latents, text_embeddings, 0.3, generator, device)
        
        real_out = model(real_images, output_features=True)
        L_real_vec = F.cross_entropy(real_out["logits"], real_labels, reduction="none").detach()
        real_feats = real_out["image_feats"].detach()
        real_regions = torch.argmin(torch.cdist(real_feats.float(), centroids_tensor.float()), dim=1).detach()

    del text_embeddings, latents, real_out, real_feats
    gc.collect()
    torch.cuda.empty_cache()

    latents_t = latents_t.detach().requires_grad_(True)
    
    mean = torch.tensor([0.481, 0.457, 0.408], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.268, 0.261, 0.275], device=device).view(1, 3, 1, 1)

    for k in range(config.train.opt_steps):
        if latents_t.grad is not None:
            latents_t.grad.zero_()

        for i in range(num_samples):
            sub_latent = latents_t[i:i+1]
            sub_label = synth_labels[i:i+1]

            with torch.amp.autocast("cuda", dtype=torch.float16):
                decoded = pipe.vae.decode(sub_latent / pipe.vae.config.scaling_factor).sample
                image = (decoded / 2 + 0.5).clamp(0, 1)
                
                image_clip = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
                image_trans = (image_clip - mean) / std
                
                out = model(image_trans, output_features=True)
                s_logits, s_feats = out["logits"], out["image_feats"]

                s_loss = F.cross_entropy(s_logits, sub_label)
                
                dist_s = torch.cdist(s_feats.float(), centroids_tensor.float())
                s_region = torch.argmin(dist_s, dim=1)
                
                eps_GS_sub = torch.tensor(0.0, device=device)
                r_mask = (real_regions == s_region)
                if r_mask.any():
                    diff = torch.abs(L_real_vec[r_mask] - s_loss)
                    eps_GS_sub = diff.mean()

                total_loss = (-config.train.lamda4 * s_loss + config.train.lamda5 * eps_GS_sub) / num_samples

            total_loss.backward()

            del decoded, image, image_clip, image_trans, out, s_logits, s_feats, total_loss
        
        with torch.no_grad():
            latents_t.data -= alpha * latents_t.grad.data
            latents_t.grad.zero_() 

        if k % 2 == 0:
            torch.cuda.empty_cache()

    with torch.no_grad():
        final_list = []
        for i in range(num_samples):
            z = latents_t[i:i+1] / pipe.vae.config.scaling_factor
            img = pipe.vae.decode(z).sample
            img_resized = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
            final_list.append((img_resized / 2 + 0.5).clamp(0, 1))
        
        final_images = torch.cat(final_list, dim=0).detach()

    return final_images