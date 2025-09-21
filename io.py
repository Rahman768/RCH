import os
import torch


def ensure_dirs(*paths: str):
    for p in paths:
        if p:
            os.makedirs(p, exist_ok=True)


@torch.no_grad()
def save_soft_targets_with_features(model: torch.nn.Module, dataloader, device: torch.device, save_path: str):
    model.eval()
    out = {}
    for i, (images, _) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        logits, high_feat, low_feats = model(images, return_features=True)
        out[f"sample_{i}"] = {
            "logits": logits.detach().cpu(),
            "high_feat": high_feat.detach().cpu(),
            "low_feats": [t.detach().cpu() for t in low_feats],
        }
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(out, save_path)


def load_soft_targets(path: str, map_location: str | torch.device = "cpu"):
    return torch.load(path, map_location=map_location)
