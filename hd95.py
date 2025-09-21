import numpy as np
import torch
from medpy.metric.binary import hd95 as _hd95

def calculate_hd95(pred: torch.Tensor, target: torch.Tensor) -> float:
    if pred.dim() == 4:
        pred = pred[0]
    if target.dim() == 4:
        target = target[0]
    if pred.dtype != torch.bool:
        pred = pred != 0
    if target.dtype != torch.bool:
        target = target != 0
    p = pred.detach().cpu().numpy().astype(bool)
    t = target.detach().cpu().numpy().astype(bool)
    if p.sum() == 0 or t.sum() == 0:
        return 0.0
    return float(_hd95(p, t))

__all__ = ["calculate_hd95"]
