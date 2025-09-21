import os
import argparse
import torch
from torch.utils.data import DataLoader
from torch.cuda import amp

from models.teachernet import TeacherNet
from data.brats_dataset import BraTSDataset, get_transforms


def export_soft_targets(
    data_dir: str,
    ckpt_path: str,
    out_path: str,
    patch_size=(128, 128, 128),
    batch_size=1,
    num_workers=0,
    use_amp=True,
    device_str=None,
):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))

    _, val_t = get_transforms(patch_size)
    ds = BraTSDataset(data_dir, transform=val_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = TeacherNet(in_channels=4, out_channels=4).to(device).eval()
    if ckpt_path and os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state if isinstance(state, dict) else state["state_dict"])

    autocast = amp.autocast(enabled=(use_amp and device.type == "cuda"))
    bank = {}
    idx_base = 0

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device, non_blocking=True)

            with autocast:
                logits, deep_feat, shallow_feats = model(images, return_features=True)

            if logits.dim() == 5 and logits.size(0) > 1:
                for b in range(logits.size(0)):
                    key = f"sample_{idx_base + b}"
                    bank[key] = {
                        "logits": logits[b].detach().cpu().float(),
                        "high_feat": deep_feat[b].detach().cpu().float(),
                        "low_feats": [sf[b].detach().cpu().float() for sf in shallow_feats],
                    }
                idx_base += logits.size(0)
            else:
                key = f"sample_{idx_base}"
                bank[key] = {
                    "logits": logits[0].detach().cpu().float(),
                    "high_feat": deep_feat[0].detach().cpu().float(),
                    "low_feats": [sf[0].detach().cpu().float() for sf in shallow_feats],
                }
                idx_base += 1

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    torch.save(bank, out_path)
    print(f"Saved {len(bank)} samples to {out_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Export Teacher logits + features for KD")
    p.add_argument("--data-dir", type=str, required=True, help="Training split directory with .h5 files")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to trained Teacher .pth")
    p.add_argument("--out", type=str, required=True, help="Output path, e.g. ./outputs/soft_targets.pkl")
    p.add_argument("--patch-size", nargs=3, type=int, default=[128, 128, 128])
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", type=str, default=None, help="cuda or cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_soft_targets(
        data_dir=args.data_dir,
        ckpt_path=args.checkpoint,
        out_path=args.out,
        patch_size=tuple(args.patch_size),
        batch_size=args.batch_size,
        num_workers=args.workers,
        use_amp=not args.no_amp,
        device_str=args.device,
    )
