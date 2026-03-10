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

def gen_hard_samples(model, pipe, synth_images_01, synth_labels, real_images, real_labels, 
                     classes, config, accelerator, centroids_tensor):
    device = accelerator.device
    alpha = 0.01 
    generator = torch.Generator(device=device).manual_seed(config.seed)

    if torch.cuda.is_available():
        try:
            pipe.unet.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"xformers is not available. ({e})")
        
        pipe.vae.enable_slicing()

    selected_style = REALISTIC_STYLE_POOL[torch.randint(0, len(REALISTIC_STYLE_POOL), (1,), generator=generator).item()]

    with torch.no_grad():
        latents, text_embeddings = precompute_assets(pipe, synth_images_01, synth_labels, classes, selected_style, device)
        
        real_out = model(real_images, output_features=True)
        L_real_vec = F.cross_entropy(real_out["logits"], real_labels, reduction="none")
        real_feats = real_out["image_feats"]
        dists_real = torch.cdist(real_feats.float(), centroids_tensor.float())
        real_regions = torch.argmin(dists_real, dim=1)

        noise_strength = 0.3
        num_inference_steps = 50
        pipe.scheduler.set_timesteps(num_inference_steps, device=device)
        init_step = int(num_inference_steps * noise_strength)
        t_start = pipe.scheduler.timesteps[-init_step]
        
        noise = torch.randn(latents.shape, generator=generator, device=device, dtype=latents.dtype)
        latents_t = pipe.scheduler.add_noise(latents, noise, torch.tensor([t_start], device=device))

        for t in pipe.scheduler.timesteps[-init_step:]:
            latent_model_input = torch.cat([latents_t] * 2)
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
            latents_t = pipe.scheduler.step(noise_pred, t, latents_t).prev_sample

    del text_embeddings, noise, noise_pred
    gc.collect()
    torch.cuda.empty_cache()

    latents_t = latents_t.detach().requires_grad_(True)
    
    mean = torch.tensor([0.481, 0.457, 0.408], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.268, 0.261, 0.275], device=device).view(1, 3, 1, 1)

    final_images = None
    for k in range(config.train.opt_steps):
        with torch.amp.autocast("cuda", dtype=torch.float16):
            decoded = pipe.vae.decode(latents_t / pipe.vae.config.scaling_factor).sample
            images = (decoded / 2 + 0.5).clamp(0, 1)
            
            images_clip = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
            images_trans = (images_clip - mean) / std

            out = model(images_trans, output_features=True)
            synth_logits, synth_feats = out["logits"], out["image_feats"]
            
            L_synth_vec = F.cross_entropy(synth_logits, synth_labels, reduction="none")
            synth_loss = L_synth_vec.mean()

            eps_GS = torch.tensor(0.0, device=device)
            g_count = 0
            dists_synth = torch.cdist(synth_feats.float(), centroids_tensor.float())
            synth_regions = torch.argmin(dists_synth, dim=1)

            for i in synth_regions.unique():
                r_mask = (real_regions == i)
                s_mask = (synth_regions == i)
                if r_mask.any():
                    diff = torch.abs(L_real_vec[r_mask].view(-1, 1) - L_synth_vec[s_mask].view(1, -1))
                    eps_GS += diff.mean() * s_mask.sum()
                    g_count += s_mask.sum()

            if g_count > 0: eps_GS /= g_count

            loss = -config.train.lamda4 * synth_loss + config.train.lamda5 * eps_GS

        loss.backward()
        
        with torch.no_grad():
            latents_t -= alpha * latents_t.grad
            latents_t.grad.zero_() 

        if k == config.train.opt_steps - 1:
            final_images = images.detach().clone()

    return final_images