import os
import argparse
import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda import amp

from models.teachernet import TeacherNet
from losses.weighted_dice import WeightedDiceLoss
from data.brats_dataset import BraTSDataset, get_transforms
from utils.ddp import setup_ddp, cleanup_ddp, is_main_process, save_on_master, get_world_size
from utils.eval import evaluate
from utils.seed import set_seed
from utils.io import ensure_dirs, save_soft_targets_with_features


class EarlyStopping:
    def __init__(self, patience=25, delta=0.0, path="checkpoint.pth"):
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
            save_on_master(model.state_dict(), self.path)
            self.counter = 0
        elif val_loss > self.best_val + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0


def build_loaders(train_dir, val_dir, patch_size, batch_size, num_workers, rank, world_size):
    train_t, val_t = get_transforms(patch_size)
    train_ds = BraTSDataset(train_dir, transform=train_t)
    val_ds = BraTSDataset(val_dir, transform=val_t)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, sampler=val_sampler,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader, train_ds


def train_one_epoch(model, loader, optimizer, scaler, criterion, device):
    model.train()
    running = 0.0
    use_amp = torch.cuda.is_available()
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(enabled=use_amp):
            logits = model(images)
            if logits.shape[2:] != masks.shape[1:]:
                logits = torch.nn.functional.interpolate(
                    logits, size=masks.shape[1:], mode="trilinear", align_corners=False
                )
            loss = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += float(loss.item())
    return running / max(1, len(loader))


def main_worker(rank, args):
    setup_ddp(rank, args.world_size, backend="nccl" if torch.cuda.is_available() else "gloo")
    set_seed(args.seed, cudnn_deterministic=True, cudnn_benchmark=True)

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    ensure_dirs(args.output_dir, os.path.dirname(args.checkpoint), os.path.dirname(args.soft_targets))

    model = TeacherNet(in_channels=4, out_channels=4).to(device)
    if torch.cuda.is_available():
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    else:
        model = DDP(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, args.t0), T_mult=1, eta_min=args.eta_min
    )
    scaler = amp.GradScaler(enabled=torch.cuda.is_available())
    criterion = WeightedDiceLoss(ce_weight=args.ce_weight, dice_weight=args.dice_weight)

    train_loader, val_loader, train_ds = build_loaders(
        args.train_dir, args.val_dir, tuple(args.patch_size), args.batch_size, args.workers, rank, args.world_size
    )

    stopper = EarlyStopping(patience=args.patience, delta=args.delta, path=args.checkpoint) if is_main_process() else None

    for epoch in range(args.epochs):
        train_loader.sampler.set_epoch(epoch)
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion, device)

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

        scheduler.step(epoch + 1e-9)

    if is_main_process() and args.dump_soft_targets:
        eval_model = model.module if isinstance(model, DDP) else model
        full_loader = DataLoader(train_ds, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
        save_soft_targets_with_features(eval_model, full_loader, device, args.soft_targets)

    cleanup_ddp()


def parse_args():
    p = argparse.ArgumentParser(description="DDP Training - TeacherNet")
    p.add_argument("--train-dir", type=str, required=True)
    p.add_argument("--val-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, default="./outputs")
    p.add_argument("--checkpoint", type=str, default="./outputs/teacher_best.pth")
    p.add_argument("--soft-targets", type=str, default="./outputs/soft_targets.pkl")
    p.add_argument("--dump-soft-targets", action="store_true")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--eta-min", type=float, default=1e-6)
    p.add_argument("--t0", type=int, default=15)
    p.add_argument("--ce-weight", type=float, default=0.3)
    p.add_argument("--dice-weight", type=float, default=0.7)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--delta", type=float, default=0.0)
    p.add_argument("--patch-size", nargs=3, type=int, default=[128, 128, 128])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--world-size", type=int, required=True)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    torch.multiprocessing.spawn(main_worker, nprocs=args.world_size, args=(args,))
