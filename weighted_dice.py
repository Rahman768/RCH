import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedDiceLoss(nn.Module):
    def __init__(self, ce_weight=0.3, dice_weight=0.7, weights=(1.64, 2.55, 3.40)):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.weights = weights

    def forward(self, outputs, targets):
        smooth = 1e-5
        ce = self.ce_loss(outputs, targets)

        probs = F.softmax(outputs, dim=1)

        wt_mask = (targets > 0).float().to(outputs.device)
        tc_mask = ((targets == 1) | (targets == 3)).float().to(outputs.device)
        et_mask = (targets == 3).float().to(outputs.device)

        wt_pred = probs[:, 1:].sum(dim=1)
        tc_pred = probs[:, 1] + probs[:, 3]
        et_pred = probs[:, 3]

        wt_inter = (wt_pred * wt_mask).sum()
        tc_inter = (tc_pred * tc_mask).sum()
        et_inter = (et_pred * et_mask).sum()

        wt_union = wt_pred.pow(2).sum() + wt_mask.pow(2).sum()
        tc_union = tc_pred.pow(2).sum() + tc_mask.pow(2).sum()
        et_union = et_pred.pow(2).sum() + et_mask.pow(2).sum()

        wt_dice = 1 - (2 * wt_inter + smooth) / (wt_union + smooth)
        tc_dice = 1 - (2 * tc_inter + smooth) / (tc_union + smooth)
        et_dice = 1 - (2 * et_inter + smooth) / (et_union + smooth)

        total_dice = self.weights[0] * wt_dice + self.weights[1] * tc_dice + self.weights[2] * et_dice
        return self.ce_weight * ce + self.dice_weight * total_dice


__all__ = ["WeightedDiceLoss"]
