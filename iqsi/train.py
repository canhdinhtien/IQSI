import torch
import torch.nn.functional as F
from .gen_hard_sample import gen_hard_samples
from torchvision.transforms import v2

def train_step(
    model, 
    real_batch, 
    synth_batch, 
    centroids_tensor, 
    optimizer, 
    accelerator, 
    config, 
    dtype_clip,
    autocast_context
):
    real_images, real_labels, _ = real_batch
    synth_images, synth_labels, _ = synth_batch
    
    TS = {i: [[], []] for i in range(config.train.num_clusters)}
    
    with autocast_context:
        real_images = real_images.to(accelerator.device, dtype=dtype_clip)
        real_labels = real_labels.to(accelerator.device)
        synth_images = synth_images.to(accelerator.device, dtype=dtype_clip)
        synth_labels = synth_labels.to(accelerator.device)

        real_out = model(real_images, output_features=True)
        synth_out = model(synth_images, output_features=True)
        
        real_logits, real_feats = real_out["logits"], real_out["image_feats"]
        synth_logits, synth_feats = synth_out["logits"], synth_out["image_feats"]

        L_real_vector = F.cross_entropy(real_logits, real_labels, reduction="none")
        L_synth_vector = F.cross_entropy(synth_logits, synth_labels, reduction="none")
        
        real_loss = L_real_vector.mean()
        synth_loss = L_synth_vector.mean()

        with torch.no_grad():
            dists_real = torch.cdist(real_feats.float(), centroids_tensor.float())
            real_regions = torch.argmin(dists_real, dim=1)
            
            dists_synth = torch.cdist(synth_feats.float(), centroids_tensor.float())
            synth_regions = torch.argmin(dists_synth, dim=1)
            
            for b_idx, r_idx in enumerate(real_regions):
                TS[r_idx.item()][0].append(b_idx)
            for b_idx, s_idx in enumerate(synth_regions):
                TS[s_idx.item()][1].append(b_idx)

        eps_GS = torch.tensor(0.0, device=accelerator.device)
        eps_GZ = torch.tensor(0.0, device=accelerator.device)
        g = 0

        for i in range(config.train.num_clusters):
            real_indices = TS[i][0]
            synth_indices = TS[i][1]
            num_s = len(synth_indices)

            if num_s > 0:
                if len(real_indices) > 0:
                    diff_gs = torch.abs(L_real_vector[real_indices].unsqueeze(1) - L_synth_vector[synth_indices].unsqueeze(0))
                    eps_GS += diff_gs.mean() * num_s
                
                if num_s > 1:
                    diff_gz = torch.abs(L_synth_vector[synth_indices].unsqueeze(1) - L_synth_vector[synth_indices].unsqueeze(0))
                    eps_GZ += diff_gz.mean() * num_s
                
                g += num_s

        if g > 0:
            eps_GS = eps_GS / g
            eps_GZ = eps_GZ / g

        total_loss = synth_loss + config.train.lamda1 * real_loss + config.train.lamda2 * eps_GS + config.train.lamda3 * eps_GZ

    optimizer.zero_grad()
    accelerator.backward(total_loss)
    optimizer.step()

    log_data = {
        "loss/total": total_loss.item(),
        "loss/real_ce": real_loss.item(),
        "loss/synth_ce": synth_loss.item(),
        "loss/eps_GS": eps_GS.item() if torch.is_tensor(eps_GS) else eps_GS,
        "loss/eps_GZ": eps_GZ.item() if torch.is_tensor(eps_GZ) else eps_GZ,
        "stats/active_clusters": sum(1 for i in range(config.train.num_clusters) if len(TS[i][1]) > 0)
    }
    
    return log_data

def train_transform_tensor(image_batch):
    augmentor = v2.Compose([
        v2.RandAugment(num_ops=2, magnitude=9),
        v2.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                     std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    return augmentor(image_batch)

def denormalize_clip(tensor):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(tensor.device).view(1, 3, 1, 1)
    
    res = (tensor * std) + mean
    return torch.clamp(res, 0, 1)

def train_step_with_hard_samples(
    model, 
    pipe, 
    real_batch, 
    synth_batch,
    centroids_tensor, 
    optimizer, 
    accelerator, 
    config, 
    dtype_clip, 
    autocast_context,
    hard_ratio,
    classes
):
    model.train()
    
    real_images, real_labels, _ = real_batch
    synth_images, synth_labels, _ = synth_batch

    bs = synth_images.shape[0]
    num_hard = int(bs * hard_ratio)

    perm = torch.randperm(bs)
    hard_indices = perm[:num_hard]
    
    images_to_transform = synth_images[hard_indices].to(accelerator.device)
    images_01 = denormalize_clip(images_to_transform)
    
    hard_samples = gen_hard_samples(
        model, 
        pipe, 
        images_01,
        synth_labels[hard_indices].to(accelerator.device), 
        real_images,
        real_labels,
        classes,
        config, 
        accelerator, 
        centroids_tensor
    )
    
    hard_samples_aug = train_transform_tensor(hard_samples).to(accelerator.device, dtype=dtype_clip)
    
    updated_synth_images = synth_images.clone().to(accelerator.device, dtype=dtype_clip)
    updated_synth_images[hard_indices] = hard_samples_aug

    TS = {i: [[], []] for i in range(config.train.num_clusters)}

    with autocast_context:
        real_images = real_images.to(accelerator.device, dtype=dtype_clip)
        real_labels = real_labels.to(accelerator.device)
        synth_labels = synth_labels.to(accelerator.device)

        real_out = model(real_images, output_features=True)
        synth_out = model(updated_synth_images, output_features=True)
        
        real_logits, real_feats = real_out["logits"], real_out["image_feats"]
        synth_logits, synth_feats = synth_out["logits"], synth_out["image_feats"]

        L_real_vector = F.cross_entropy(real_logits, real_labels, reduction="none")
        L_synth_vector = F.cross_entropy(synth_logits, synth_labels, reduction="none")
        
        real_loss = L_real_vector.mean()
        synth_loss = L_synth_vector.mean()

        with torch.no_grad():
            real_regions = torch.argmin(torch.cdist(real_feats.float(), centroids_tensor.float()), dim=1)
            synth_regions = torch.argmin(torch.cdist(synth_feats.float(), centroids_tensor.float()), dim=1)
            
            for b_idx, r_idx in enumerate(real_regions):
                TS[r_idx.item()][0].append(b_idx)
            for b_idx, s_idx in enumerate(synth_regions):
                TS[s_idx.item()][1].append(b_idx)

        eps_GS = torch.tensor(0.0, device=accelerator.device)
        eps_GZ = torch.tensor(0.0, device=accelerator.device)
        g = 0

        for i in range(config.train.num_clusters):
            r_idx, s_idx = TS[i][0], TS[i][1]
            num_s = len(s_idx)
            if num_s > 0:
                if len(r_idx) > 0:
                    diff_gs = torch.abs(L_real_vector[r_idx].unsqueeze(1) - L_synth_vector[s_idx].unsqueeze(0))
                    eps_GS += diff_gs.mean() * num_s
                if num_s > 1:
                    diff_gz = torch.abs(L_synth_vector[s_idx].unsqueeze(1) - L_synth_vector[s_idx].unsqueeze(0))
                    eps_GZ += diff_gz.mean() * num_s
                g += num_s

        if g > 0:
            eps_GS /= g
            eps_GZ /= g

        total_loss = synth_loss + config.train.lamda1 * real_loss + config.train.lamda2 * eps_GS + config.train.lamda3 * eps_GZ

    optimizer.zero_grad()
    accelerator.backward(total_loss)
    optimizer.step()

    return {
        "loss/total": total_loss.item(),
        "loss/real_ce": real_loss.item(),
        "loss/synth_ce": synth_loss.item(),
        "loss/eps_GS": eps_GS.item(),
        "loss/eps_GZ": eps_GZ.item(),
        "stats/active_clusters": sum(1 for i in range(config.train.num_clusters) if len(TS[i][1]) > 0)
    }