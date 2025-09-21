# RMM: Teacher–Student Framework for Brain Tumor Segmentation

## Implementation Details

We adopted a transformer–convolution hybrid architecture for the Teacher Model, incorporating multiple attention mechanisms and advanced skip connections.
The Student Model was designed with a lightweight structure using Conv3D, ECA, and a KAN block to reduce computational load while maintaining segmentation quality.

All experiments were conducted on a high-performance system with:
- Intel Core i9-14900K CPU  
- 48 GB RAM  
- NVIDIA RTX 4090 GPUs  

The implementation was developed in **PyTorch** with **CUDA** support. Preprocessing and data augmentation were carried out using the **TorchIO** library.

To improve training efficiency:
- Mixed-precision training was applied  
- Gradient checkpointing was enabled in selected Teacher Model layers  
- Teacher Model was trained using **Distributed Data Parallel (DDP)** across multiple GPUs  
- Student Model was trained on a single GPU  
## Project Structure
├── models/
│ ├── teachernet.py
│ └── studentnet.py
├── losses/
│ └── weighted_dice.py
├── metrics/
│ └── hd95.py
├── data/
│ └── brats_dataset.py
├── utils/
│ ├── ddp.py
│ ├── eval.py
│ ├── seed.py
│ └── io.py
├── preprocess/
│ ├── bias_correction.py
│ └── preprocess_brats.py
├── train.py
├── train_student.py
├── save_kd_targets.py
├── eval.py
├── infer.py
└── README.md


## Requirements
- Python 3.9+
- PyTorch 2.0+
- TorchIO
- SimpleITK
- nibabel
- medpy
- h5py

Citation
@article{rahman2025swinunet,
  title={Advancedly Automatic Brain Tumor Segmentation Utilizing SwinTunet and Cross-Attention},
  author={Rahman, Mostafizur and Sun, Ke and Wang, Yu},
  journal={Biomedical Signal Processing and Control},
  year={2025}
}


