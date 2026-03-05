import torch
from tqdm import tqdm

@torch.no_grad()
def evaluate(model, data_loader, accelerator):
    model.eval()
    all_preds = []
    all_labels = []

    for batch in tqdm(data_loader, desc="Evaluating", disable=not accelerator.is_main_process):
        images = batch[0].to(accelerator.device)
        labels = batch[1].to(accelerator.device)
        
        with accelerator.autocast():
            logits = model(images)
        
        preds = torch.argmax(logits, dim=1)
        
        all_preds.append(accelerator.gather(preds))
        all_labels.append(accelerator.gather(labels))

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    
    acc = (all_preds == all_labels).float().mean().item()
    return acc