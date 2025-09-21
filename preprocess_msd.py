import os
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import h5py


def image_resampling(image, new_shape, order=3):
    image_tensor = torch.tensor(image, dtype=torch.float32, device="cuda")
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    resampled = F.interpolate(image_tensor, size=new_shape, mode="trilinear" if order > 0 else "nearest", align_corners=False)
    return resampled.squeeze().cpu().numpy()


def intensity_normalization(image):
    image_tensor = torch.tensor(image, dtype=torch.float32, device="cuda")
    min_val, max_val = torch.min(image_tensor), torch.max(image_tensor)
    if min_val == max_val:
        return torch.ones_like(image_tensor).cpu().numpy()
    return ((image_tensor - min_val) / (max_val - min_val)).cpu().numpy()


def remap_labels(segmentation):
    out = np.zeros_like(segmentation, dtype=np.uint8)
    out[segmentation == 1] = 1  # NCR/NET
    out[segmentation == 2] = 2  # ED
    out[segmentation == 4] = 3  # ET
    return out


def preprocess_data(data_path, output_path, new_shape=(128, 128, 128)):
    os.makedirs(output_path, exist_ok=True)
    patients = [p for p in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, p))]

    for pid in patients:
        pdir = os.path.join(data_path, pid)
        seg = nib.load(os.path.join(pdir, f"{pid}_seg.nii.gz")).get_fdata().astype(np.uint8)
        t1ce = nib.load(os.path.join(pdir, f"{pid}_t1ce.nii.gz")).get_fdata()
        t1 = nib.load(os.path.join(pdir, f"{pid}_t1.nii.gz")).get_fdata()
        flair = nib.load(os.path.join(pdir, f"{pid}_flair.nii.gz")).get_fdata()
        t2 = nib.load(os.path.join(pdir, f"{pid}_t2.nii.gz")).get_fdata()

        t1ce = intensity_normalization(image_resampling(t1ce, new_shape))
        t1 = intensity_normalization(image_resampling(t1, new_shape))
        flair = intensity_normalization(image_resampling(flair, new_shape))
        t2 = intensity_normalization(image_resampling(t2, new_shape))
        seg = remap_labels(image_resampling(seg, new_shape, order=0))

        combined = np.stack([t1ce, t1, flair, t2], axis=0)

        ofile = os.path.join(output_path, f"{pid}.h5")
        with h5py.File(ofile, "w") as hf:
            hf.create_dataset("image", data=combined.astype(np.float32), compression="gzip")
            hf.create_dataset("label", data=seg.astype(np.uint8), compression="gzip")

        print("Saved", ofile)


if __name__ == "__main__":
    data_path = r"F:\Research\Dataset\Brats 2021\RSNA_ASNR_MICCAI_BraTS2021_TrainingData_16July2021"
    output_path = r"F:\Research\Preprocess 2021\128"
    preprocess_data(data_path, output_path)
