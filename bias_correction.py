import os
import nibabel as nib
import SimpleITK as sitk
import numpy as np


def bias_field_correction(image: np.ndarray) -> np.ndarray:
    sitk_image = sitk.GetImageFromArray(image.astype(np.float32))
    mask_image = sitk.OtsuThreshold(sitk_image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_image = corrector.Execute(sitk_image, mask_image)
    return sitk.GetArrayFromImage(corrected_image)


def run_n4_batch(input_root: str, output_root: str, modalities=("flair", "t1", "t1ce", "t2")):
    os.makedirs(output_root, exist_ok=True)
    for folder in os.listdir(input_root):
        folder_path = os.path.join(input_root, folder)
        if not os.path.isdir(folder_path):
            continue

        out_folder = os.path.join(output_root, folder)
        os.makedirs(out_folder, exist_ok=True)

        for mod in modalities:
            fname = f"{folder}_{mod}.nii.gz"
            in_path = os.path.join(folder_path, fname)
            if not os.path.exists(in_path):
                print(f"Missing: {in_path}")
                continue

            img_nib = nib.load(in_path)
            data = img_nib.get_fdata()
            corrected = bias_field_correction(data)

            out_path = os.path.join(out_folder, fname)
            nib.save(nib.Nifti1Image(corrected, affine=img_nib.affine, header=img_nib.header), out_path)
            print(f"Saved: {out_path}")


if __name__ == "__main__":
    input_root = r"url "
    output_root = r"url"
    run_n4_batch(input_root, output_root)
