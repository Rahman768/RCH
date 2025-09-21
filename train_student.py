import os
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda import amp
import pickle

from models.studentnet import StudentNet
from data.brats_dataset import BraTSDataset, get_transforms
from utils.ddp import setup_ddp, cleanup_ddp, is_main_process, get_world_size
from utils.eval import evaluate
from utils.seed import set_seed
from utils.io import ensure_dirs


class EarlyStopping:
    def __init__(self, patience=25, delta=0.0, path="student_best.pth"):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_dice = None
        self.best_val = float("inf")
        self.early_stop = False

    def __call__(self, val_loss, et, tc, wt, model):
        avg_dice = (et + tc + wt) / 3.0
        if self.best_dice is None or avg_dice > self.best_dice + self.delta:
            self.best_dice = avg_dice
            self.best_val = val_loss
            torch.save(model.state_dict(), self.path)
            self.counter = 0
        elif val_loss > self.best_val + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0


class DistillationLoss(torch.nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, temperature=2.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature
        self.mse = torch.nn.MSELoss()
        self.kld = torch.nn.KLDivLoss(reduction="batchmean")

    @staticmethod
    def _align_spatial(t: torch.Tensor, target_spatial):
        if t.shape[2:] != target_spatial:
            t = F.interpolate(t, size=target_spatial, mode="trilinear", align_corners=False)
        return t

    @staticmethod
    def _match_channels(t: torch.Tensor, target_ch: int):
        c = t.shape[1]
        if c == target_ch:
            return t
        if c > target_ch:
            return t[:, :target_ch, ...]
        pad_ch = target_ch - c
        pad = (0, 0, 0, 0, 0, 0, 0, pad_ch)
        return F.pad(t, pad, mode="constant", value=0)

    def forward(self, s_out, gt_mask, s_deep, s_shallow, t_out, t_deep, t_shallow):
        if gt_mask.shape[1:] != s_out.shape[2:]:
            gt_mask = F.interpolate(gt_mask.unsqueeze(1).float(), size=s_out.shape[2:], mode="nearest").squeeze(1).long()
        ce = F.cross_entropy(s_out, gt_mask)

        if t_out.shape[2:] != s_out.shape[2:]:
            t_out = F.interpolate(t_out, size=s_out.shape[2:], mode="trilinear", align_corners=False)
        s_logp = F.log_softmax(s_out / self.temperature, dim=1)
        t_prob = F.softmax(t_out / self.temperature, dim=1)
        l_kd = self.kld(s_logp, t_prob) * (self.temperature ** 2)

        t_deep = self._align_spatial(t_deep, s_deep.shape[2:])
        t_deep = self._match_channels(t_deep, s_deep.shape[1])
        l_cd = self.mse(s_deep, t_deep)

        n = min(len(s_shallow), len(t_shallow))
        for i in range(n):
            ts = self._align_spatial(t_shallow[i], s_shallow[i].shape[2:])
            ts = self._match_channels(ts, s_shallow[i].shape[1])
            l_cd = l_cd + self.mse(s_shallow[i], ts)

        total = ce + self.alpha * l_cd + self.beta * l_kd
        return total, ce.item(), l_cd.item(), l_kd.item()


def build_loaders(train_dir, val_dir, patch_size, batch_size, num_workers, rank, world_size, shuffle_kd):
    train_t, val_t = get_transforms(patch_size)
    train_ds = BraTSDataset(train_dir, transform=train_t)
    val_ds = BraTSDataset(val_dir, transform=val_t)

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=shuffle_kd, drop_last=False
    )
    val_sampler = DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, sampler=val_sampler,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader


def load_soft_targets(path, map_location="cpu"):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        return obj
    try:
        return torch.load(path, map_location=map_location)
    except Exception:
        return obj


def train_one_epoch(model, loader, optimizer, scaler, kd_criterion, device, teacher_bank, require_ordered, temperature):
    model.train()
    running = 0.0
    use_amp = torch.cuda.is_available()

    if require_ordered:
        assert hasattr(loader.sampler, "indices"), "DistributedSampler must expose indices when shuffle=False"

    for step, (images, gt_masks) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        gt_masks = gt_masks.to(device, non_blocking=True)

        if require_ordered:
            if hasattr(loader.sampler, "indices"):
                ds_idx = loader.sampler.indices[step]
            else:
                ds_idx = step
            key = f"sample_{ds_idx}"
        else:
            key = f"sample_{step}"

        if key not in teacher_bank:
            continue

        t_entry = teacher_bank[key]
        t_out = t_entry["logits"].to(device, non_blocking=True)
        t_deep = t_entry["high_feat"].to(device, non_blocking=True)
        t_shallow = [feat.to(device, non_blocking=True) for feat in t_entry["low_feats"]]

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(enabled=use_amp):
            s_out, s_deep, s_shallow = model(images, return_features=True)
            loss, ce_v, cd_v, kd_v = kd_criterion(
                s_out, gt_masks, s_deep, s_shallow, t_out, t_deep, t_shallow
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += float(loss.item())

        if step % 10 == 0:
            print(f"Step {step:05d} | Loss {loss.item():.4f} | CE {ce_v:.4f} | CD {cd_v:.4f} | KD {kd_v:.4f}")

    return running / max(1, len(loader))


def main_worker(rank, args):
    setup_ddp(rank, args.world_size, backend="nccl" if torch.cuda.is_available() else "gloo")
    set_seed(args.seed, cudnn_deterministic=True, cudnn_benchmark=True)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    ensure_dirs(args.output_dir, os.path.dirname(args.checkpoint))

    teacher_bank = load_soft_targets(args.soft_targets, map_location="cpu")

    shuffle_kd = args.shuffle_kd
    if shuffle_kd and is_main_process():
        print("Warning: shuffle_kd=True can break teacher-student alignment; prefer False with batch_size=1.")

    train_loader, val_loader = build_loaders(
        args.train_dir, args.val_dir, tuple(args.patch_size), args.batch_size, args.workers, rank, args.world_size, shuffle_kd
    )

    model = StudentNet(in_channels=4, out_channels=4).to(device)
    if torch.cuda.is_available():
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    else:
        model = DDP(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.tmax), eta_min=args.eta_min)
    scaler = amp.GradScaler(enabled=torch.cuda.is_available())
    kd_criterion = DistillationLoss(alpha=args.alpha, beta=args.beta, temperature=args.temperature)

    stopper = EarlyStopping(patience=args.patience, delta=args.delta, path=args.checkpoint) if is_main_process() else None

    require_ordered = (not shuffle_kd) and (args.batch_size == 1)

    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch)
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, kd_criterion, device, teacher_bank, require_ordered, args.temperature)

        if is_main_process():
            eval_model = model.module if isinstance(model, DDP) else model
            metrics = evaluate(eval_model, val_loader, device=device)
            print(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"TrainLoss={train_loss:.4f} | "
                f"ValLoss={metrics['val_loss']:.4f} | "
                f"ET={metrics['et_dice']:.4f} | TC={metrics['tc_dice']:.4f} | WT={metrics['wt_dice']:.4f}"
            )
            stopper(metrics["val_loss"], metrics["et_dice"], metrics["tc_dice"], metrics["wt_dice"], eval_model)
            if stopper.early_stop:
                print("Early stopping triggered.")
                break

        scheduler.step()

    cleanup_ddp()


def parse_args():
    p = argparse.ArgumentParser(description="DDP Training - StudentNet with KD")
    p.add_argument("--train-dir", type=str, required=True)
    p.add_argument("--val-dir", type=str, required=True)
    p.add_argument("--soft-targets", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="./outputs")
    p.add_argument("--checkpoint", type=str, default="./outputs/student_best.pth")

    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--workers", type=int, default=0)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--eta-min", type=float, default=1e-6)
    p.add_argument("--tmax", type=int, default=500 // 33)

    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--temperature", type=float, default=2.0)

    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--delta", type=float, default=0.0)

    p.add_argument("--patch-size", nargs=3, type=int, default=[128, 128, 128])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--world-size", type=int, required=True)

    p.add_argument("--shuffle-kd", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.multiprocessing.spawn(main_worker, nprocs=args.world_size, args=(args,))
