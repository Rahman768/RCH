from typing import List, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class ECA3DBlock(nn.Module):
    def __init__(self, channels: int, k_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = y.squeeze(-1).squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)
        return x * y.expand_as(x)


class KANBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.kan = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.kan(x)


patch_aggregation = lambda x: F.avg_pool3d(x, kernel_size=2, stride=2)

def patch_merging(x: torch.Tensor) -> torch.Tensor:
    if min(x.shape[2:]) >= 2:
        return F.avg_pool3d(x, kernel_size=2, stride=2)
    else:
        return x


class StudentNet(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 4):
        super().__init__()
        channels = [32, 64, 128, 256]
        self.patch_agg = patch_aggregation
        self.encoder_stages = nn.ModuleList()
        self.encoder_ecas = nn.ModuleList()
        c_in = in_channels
        for ch in channels:
            self.encoder_stages.append(
                nn.Sequential(
                    nn.Conv3d(c_in, ch, kernel_size=3, padding=1),
                    nn.BatchNorm3d(ch),
                    nn.ReLU(inplace=True),
                )
            )
            self.encoder_ecas.append(ECA3DBlock(ch))
            c_in = ch
        self.kan = KANBlock(channels[-1])
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose3d(channels[i], channels[i - 1], kernel_size=2, stride=2)
            for i in range(len(channels) - 1, 0, -1)
        ])
        self.decoder_stages = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(channels[i - 1] * 2, channels[i - 1], kernel_size=3, padding=1),
                nn.BatchNorm3d(channels[i - 1]),
                nn.ReLU(inplace=True),
                ECA3DBlock(channels[i - 1]),
            )
            for i in range(len(channels) - 1, 0, -1)
        ])
        self.final_conv = nn.Conv3d(channels[0], out_channels, kernel_size=1)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]]:
        skips: List[torch.Tensor] = []
        shallow_feats: List[torch.Tensor] = []
        x = self.patch_agg(x)
        for conv, eca in zip(self.encoder_stages, self.encoder_ecas):
            x = conv(x)
            x = eca(x)
            skips.append(x)
            shallow_feats.append(x)
            x = patch_merging(x)
        x = self.kan(x)
        deep_feat = x
        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            skip = skips[-(i + 2)]
            if x.shape[-3:] != skip.shape[-3:]:
                skip = F.interpolate(skip, size=x.shape[-3:], mode='trilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self.decoder_stages[i](x)
        logits = self.final_conv(x)
        if return_features:
            return logits, deep_feat, shallow_feats
        else:
            return logits


__all__ = [
    "StudentNet",
    "ECA3DBlock",
    "KANBlock",
    "patch_aggregation",
    "patch_merging",
]
