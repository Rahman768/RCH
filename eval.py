import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda import amp
from metrics.hd95 import calculate_hd95


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred = (pred > threshold).float()
    target = target.float()
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (2.0 * inter / (union + 1e-6)).mean().item()


@torch.no_grad()
def evaluate(model: torch.nn.Module, val_loader, device: torch.device = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    ce = torch.nn.CrossEntropyLoss()
    val_loss = 0.0
    et_d, tc_d, wt_d = [], [], []
    et_h, tc_h, wt_h = [], [], []
    use_amp = torch.cuda.is_available()
    for images, masks in val_loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with amp.autocast(enabled=use_amp):
            logits = model(images)
            if logits.shape[2:] != masks.shape[1:]:
                logits = F.interpolate(logits, size=masks.shape[1:], mode="trilinear", align_corners=False)
            loss = ce(logits, masks)
        val_loss += float(loss.item())
        pred = torch.argmax(logits, dim=1)
        et_pred = (pred == 3).float()
        et_tgt = (masks == 3).float()
        tc_pred = ((pred == 1) | (pred == 3)).float()
        tc_tgt = ((masks == 1) | (masks == 3)).float()
        wt_pred = (pred > 0).float()
        wt_tgt = (masks > 0).float()
        et_d.append(dice_coefficient(et_pred, et_tgt))
        tc_d.append(dice_coefficient(tc_pred, tc_tgt))
        wt_d.append(dice_coefficient(wt_pred, wt_tgt))
        et_h.append(calculate_hd95(et_pred[0], et_tgt[0]))
        tc_h.append(calculate_hd95(tc_pred[0], tc_tgt[0]))
        wt_h.append(calculate_hd95(wt_pred[0], wt_tgt[0]))
    n = max(1, len(val_loader))
    return {
        "val_loss": val_loss / n,
        "et_dice": float(np.mean(et_d)) if et_d else 0.0,
        "tc_dice": float(np.mean(tc_d)) if tc_d else 0.0,
        "wt_dice": float(np.mean(wt_d)) if wt_d else 0.0,
        "et_hd95": float(np.mean(et_h)) if et_h else 0.0,
        "tc_hd95": float(np.mean(tc_h)) if tc_h else 0.0,
        "wt_hd95": float(np.mean(wt_h)) if wt_h else 0.0,
    }
