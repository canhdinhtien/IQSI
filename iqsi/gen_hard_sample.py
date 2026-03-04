import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from PIL import Image
import torchvision.transforms.functional as TF

def load_sd_images(image_paths, device="cuda", dtype=torch.float32):
    batch_tensors = []
    for p in image_paths:
        with Image.open(p).convert("RGB") as img:
            img = img.resize((224, 224), Image.BICUBIC)
            tensor = TF.to_tensor(img) 
            tensor = tensor * 2 - 1
            batch_tensors.append(tensor)
    
    return torch.stack(batch_tensors).to(device=device, dtype=dtype)

def clean_transform_tensor(image_batch):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1).to(image_batch.device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1).to(image_batch.device)
    
    if image_batch.shape[-1] != 224:
        image_batch = F.interpolate(image_batch, size=(224, 224), mode='bilinear', align_corners=False)
        
    return (image_batch - mean) / std

@torch.no_grad()
def gen_hard_samples(model, pipe, labels_rand, paths_rand, config, accelerator, dtype_clip, alpha=0.01, opt_steps=10, noise_epsilon=0.05):
    model.eval()
    vae_dtype = pipe.vae.dtype
    
    synth_images_raw = load_sd_images(paths_rand, device=accelerator.device, dtype=vae_dtype)

    latents = pipe.vae.encode(synth_images_raw).latent_dist.sample() * pipe.vae.config.scaling_factor
    latents = latents + torch.randn_like(latents) * noise_epsilon
    
    latents_t = latents.detach().clone().requires_grad_(True)
    
    model.train() 
    for k in range(opt_steps):
        with torch.amp.autocast('cuda'):
            decoded = pipe.vae.decode(latents_t / pipe.vae.config.scaling_factor).sample
            images = (decoded / 2 + 0.5).clamp(0, 1)
            
            images_model = F.interpolate(images, size=(224, 224), mode='bilinear')
            images_model = clean_transform_tensor(images_model)

            logits = model(images_model)["logits"]
            synth_loss = F.cross_entropy(logits, labels_rand)

            loss = -config.train.lamda4 * synth_loss 

        grad = torch.autograd.grad(loss, latents_t)[0]
        with torch.no_grad():
            latents_t -= alpha * grad.sign()
        latents_t.grad = None

    return ((pipe.vae.decode(latents_t.detach() / pipe.vae.config.scaling_factor).sample / 2) + 0.5).clamp(0, 1)