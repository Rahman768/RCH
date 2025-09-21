import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchio as tio


class_mapping = {0: 0, 1: 1, 2: 2, 4: 3, 3: 3}


def remap_labels(label: torch.Tensor) -> torch.Tensor:
    if label.dtype != torch.int64:
        label = label.long()
    label_np = label.cpu().numpy()
    remapped = np.vectorize(lambda v: class_mapping.get(int(v), 0))(label_np)
    return torch.from_numpy(remapped).long()


class BraTSDataset(Dataset):
    def __init__(self, data_dir: str, transform=None):
        self.file_list = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".h5")]
        self.file_list.sort()
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        with h5py.File(path, "r") as f:
            image = torch.tensor(f["image"][:].astype(np.float32))   # (C,D,H,W)
            label = torch.tensor(f["label"][:].astype(np.uint8))     # (D,H,W)
        label = remap_labels(label)
        if self.transform:
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=image),
                label=tio.LabelMap(tensor=label.unsqueeze(0)),
            )
            t = self.transform(subject)
            image = t.image.data
            label = t.label.data.squeeze(0).long()
        return image, label


def get_transforms(patch_size=(128, 128, 128)):
    train_t = tio.Compose([
        tio.CropOrPad(patch_size),
        tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),
        tio.RandomNoise(std=0.01),
        tio.RandomAffine(scales=1, degrees=10),
    ])
    val_t = tio.Compose([
        tio.CropOrPad(patch_size),
    ])
    return train_t, val_t


def make_loaders(
    train_dir: str,
    val_dir: str,
    batch_size: int = 1,
    num_workers: int = 0,
    patch_size=(128, 128, 128),
    shuffle_train: bool = True,
):
    train_t, val_t = get_transforms(patch_size)
    train_ds = BraTSDataset(train_dir, transform=train_t)
    val_ds = BraTSDataset(val_dir, transform=val_t)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


__all__ = ["BraTSDataset", "class_mapping", "remap_labels", "get_transforms", "make_loaders"]
