import torch
import faiss
import numpy as np
from PIL import Image

@torch.no_grad()
def get_centroids_from_loader(model, paths, n_clusters, clean_transform, batch_size, accelerator, dtype_clip):
    all_embeddings = []
    
    for i in range(0, len(paths), batch_size):
        chunk_paths = paths[i : i + batch_size]
        batch_tensors = []
        
        for p in chunk_paths:
            try:
                img = Image.open(p).convert('RGB')
                batch_tensors.append(clean_transform(img))
            except Exception as e:
                print(f"Lỗi khi nạp ảnh {p}: {e}")
                continue
        
        if not batch_tensors:
            continue

        batch_input = torch.stack(batch_tensors).to(accelerator.device, dtype=dtype_clip)

        image_feats = model.forward_image(batch_input)
        all_embeddings.append(image_feats.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0).float().numpy()
    
    d_model = all_embeddings.shape[1]
    kmeans = faiss.Kmeans(d_model, n_clusters, niter=20, verbose=True, gpu=False)
    kmeans.train(all_embeddings)

    return kmeans.centroids

def assign_to_regions(features, centroids):
    distances = torch.cdist(features.float(), centroids.float(), p=2)
    return torch.argmin(distances, dim=1)