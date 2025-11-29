# Enhanced CT-CBCT Image Registration for Orthopedic Surgery

### Integrating Rigid–Elastic Motion Models

This repository provides the official implementation of **“Enhanced CT-CBCT Image Registration for Orthopedic Surgery: Integrating Rigid-Elastic Motion Models.”**
 The framework integrates rigid priors derived from bony structures with an elastic deformation network to improve CT-CBCT registration accuracy, particularly in orthopedic surgical scenarios.

------

## 🌟 Key Features

- **Rigid Alignment Module (RA Module)**
   Implemented in `rigid_disp.py`.
   Generates rigid displacement fields for bony regions and serves as prior knowledge for the registration network.
- **Elastic Deformation Module (ED Module)**
   Implemented in `model.py`.
   Integrates the rigid displacement prior with image features to generate the full-image elastic deformation field for anatomically guided CT–CBCT registration.
- **Bone Shape Preservation (BSP Loss)**
   implemented in `IC.py` and `DC.py`.
   Enforces bone-volume consistency and constrains the deformation field to maintain anatomical plausibility.
- **Rigid–Elastic Multi-Stage Framework**
   Rigid priors are concatenated with learned features, guiding subsequent elastic deformation and improving overall registration quality.

------

## 📦 Repository Structure

```
.
├── train.py                 # Main training script
├── train_rigloss.py         # Fine-tuning with rigid-aware loss
├── test.py                  # Testing script
│
├── model.py                 # Main registration model (ED Module)
├── rigid_disp/              # Rigid-Aware Module
│
├── IC/                      # BSP Loss (incompressibility constraint branch)
│
├── DC/                      # BSP Loss (distance constraint branch)
│
├── data/                	 # Data loading utilities
├── README.md
└── requirements.txt
```

------

## 🔧 Installation

```
git clone https://github.com/Zq-Huang/RE-Reg.git
cd RE-Reg
pip install -r requirements.txt
```

Recommended environment: **Python 3.8+**, **PyTorch 1.12+**, CUDA-enabled GPU.

------

## 📁 Data Preparation

Organize CT and CBCT images in the following structure:

```
data/
├── train/
│   ├── 001/
│   │   ├── ct.nii.gz              # Fixed CT
│   │   ├── cbct.nii.gz            # Moving CBCT
│   │   ├── cbct_seg.nii.gz        # CBCT segmentation
│   │   └── ct_bone_labels.nii.gz  # CT bone labels (used by RA + BSP loss)
│   ├── 002/
│   └── ...
│
├── val/
│   ├── 001/
│   │   ├── ct.nii.gz
│   │   ├── cbct.nii.gz
│   │   ├── cbct_seg.nii.gz
│   │   ├── ct_total_seg.nii.gz    # Total CT segmentation (evaluation only)
│   │   └── cbct_total_seg.nii.gz  # Total CBCT segmentation (evaluation only)
│   ├── 002/
│   └── ...
│
└── test/
    ├── 001/
    │   ├── ct.nii.gz
    │   ├── cbct.nii.gz
    │   ├── cbct_seg.nii.gz
    │   ├── ct_total_seg.nii.gz
    │   └── cbct_total_seg.nii.gz
    ├── 002/
    └── ...

```

All images should be normalized to the range **[0, 1]**.

------

## 📜 Citation

If you use this code or the proposed method, please cite:

```
@article{huang2025RE-reg,
  title={Enhanced CT-CBCT Image Registration for Orthopedic Surgery: Integrating Rigid-Elastic Motion Models},
  author={Huang, Zhiqi Zhiqi Huang, Deqiang Xiao,*, Hongxun Liu, Long Shao, Danni Ai, Jingfan Fan, Tianyu Fu, Yucong Lin, Hong Song and Jian Yang},
  year={2025}
}
```

------

## 🙏 Acknowledgements

This work is heavily based on the following open-source projects, and we sincerely thank the authors for making their code publicly available:

- [**LapIRN**](https://github.com/cwmok/LapIRN)
- [**spine-ct-mr-registration**](https://github.com/BailiangJ/spine-ct-mr-registration.git)

Their contributions provided an essential foundation for the development and refinement of this repository.