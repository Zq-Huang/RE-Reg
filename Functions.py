import numpy as np
import nibabel as nib
import torch
import os, glob 

def dice(im1, atlas):
    unique_class = np.unique(atlas)
    dice = 0
    num_count = 0

    for i in unique_class:
        if (i == 0) or ((im1 == i).sum() == 0) or ((atlas == i).sum() == 0):
            continue

        sub_dice = np.sum(atlas[im1 == i] == i) * 2.0 / (np.sum(im1 == i) + np.sum(atlas == i) + 1e-5)
        sub_dice = np.round(sub_dice, 8)
        dice += sub_dice
        num_count += 1
    return dice / num_count



def generate_grid(imgshape):
    x = np.arange(imgshape[0])
    y = np.arange(imgshape[1])
    z = np.arange(imgshape[2])
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid, 0, 2)
    grid = np.swapaxes(grid, 1, 2)
    return grid


def generate_grid_unit(imgshape):
    x = (np.arange(imgshape[0]) - ((imgshape[0] - 1) / 2)) / (imgshape[0] - 1) * 2
    y = (np.arange(imgshape[1]) - ((imgshape[1] - 1) / 2)) / (imgshape[1] - 1) * 2
    z = (np.arange(imgshape[2]) - ((imgshape[2] - 1) / 2)) / (imgshape[2] - 1) * 2
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid, 0, 2)
    grid = np.swapaxes(grid, 1, 2)
    return grid


def transform_unit_flow_to_flow(flow):
    x, y, z, _ = flow.shape
    flow[:, :, :, 0] = flow[:, :, :, 0] * (z-1)/2
    flow[:, :, :, 1] = flow[:, :, :, 1] * (y-1)/2
    flow[:, :, :, 2] = flow[:, :, :, 2] * (x-1)/2

    return flow


def transform_unit_flow_to_flow_cuda(flow):
    b, x, y, z, c = flow.shape
    flow[:, :, :, :, 0] = flow[:, :, :, :, 0] * (z-1)/2
    flow[:, :, :, :, 1] = flow[:, :, :, :, 1] * (y-1)/2
    flow[:, :, :, :, 2] = flow[:, :, :, :, 2] * (x-1)/2

    return flow


def load_4D(name):
    X = nib.load(name)
    X = X.get_fdata()
    X = np.reshape(X, (1,) + X.shape)
    return X


def load_5D(name):
    X = fixed_nii = nib.load(name)
    X = X.get_fdata()
    X = np.reshape(X, (1,) + (1,) + X.shape)
    return X


def imgnorm(img):
    max_v = np.max(img)
    min_v = np.min(img)

    norm_img = (img - min_v) / (max_v - min_v)
    return norm_img


def save_img(I_img,savename,header=None,affine=None):
    if header is None or affine is None:
        affine = np.diag([1, 1, 1, 1])
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    else:
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=header)

    nib.save(new_img, savename)


def save_img_nii(I_img, savename):
    affine = np.diag([1, 1, 1, 1])
    new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    nib.save(new_img, savename)


def save_flow(I_img,savename,header=None,affine=None):
    if header is None or affine is None:
        affine = np.diag([1, 1, 1, 1])
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    else:
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=header)

    nib.save(new_img, savename)

class Dataset_train_trans():
    def __init__(self, fixed_files, moving_files, fixed_val_files, moving_val_files):
        self.fixed_files = fixed_files
        self.moving_files = moving_files
        self.fixed_val_files = fixed_val_files
        self.moving_val_files = moving_val_files


    def __len__(self):

        return len(self.fixed_files)

    def __getitem__(self, index):

        fixed_img = nib.load(self.fixed_files[index]).get_fdata()[np.newaxis, ...]
        moving_img = nib.load(self.moving_files[index]).get_fdata()[np.newaxis, ...]
        fixed_mask = nib.load(self.fixed_val_files[index]).get_fdata()[np.newaxis, ...]
        moving_mask = nib.load(self.moving_val_files[index]).get_fdata()[np.newaxis, ...]

        return fixed_img , moving_img, fixed_mask, moving_mask


def train_data(val_file):
    fixed_files = []
    moving_files = []
    fixed_mask_files = []
    moving_mask_files = []

    val_folders = glob.glob(os.path.join(val_file,  '*'))

    for folder in val_folders:
        fixed_files.extend(glob.glob(os.path.join(folder, 'cbct.nii.gz')))
        moving_files.extend(glob.glob(os.path.join(folder, 'ct.nii.gz')))
        fixed_mask_files.extend(glob.glob(os.path.join(folder, 'cbct_seg.nii.gz')))
        moving_mask_files.extend(glob.glob(os.path.join(folder, 'ct_bone_labels.nii.gz')))



    DS_VAL = Dataset_train_trans(fixed_files, moving_files, fixed_mask_files, moving_mask_files)

    return DS_VAL



def val_dataset(val_file):
    fixed_files = []
    moving_files = []
    fixed_mask_files = []
    moving_mask_files = []
    fixed_labels_files = []


    val_folders = glob.glob(os.path.join(val_file,  '*'))

    for folder in val_folders:
        fixed_files.extend(glob.glob(os.path.join(folder, 'cbct.nii.gz')))
        moving_files.extend(glob.glob(os.path.join(folder, 'ct.nii.gz')))
        fixed_mask_files.extend(glob.glob(os.path.join(folder, 'cbct_seg.nii.gz')))
        moving_mask_files.extend(glob.glob(os.path.join(folder, 'ct_total_seg.nii.gz')))
        fixed_labels_files.extend(glob.glob(os.path.join(folder, 'cbct_total_seg.nii.gz')))


    DS_VAL = Dataset_train(fixed_files, moving_files, fixed_mask_files, moving_mask_files, fixed_labels_files)

    return DS_VAL, fixed_files


class Dataset_train():
    def __init__(self, fixed_files, moving_files, fixed_val_files, moving_val_files, fixed_labels):
        self.fixed_files = fixed_files
        self.moving_files = moving_files
        self.fixed_val_files = fixed_val_files
        self.moving_val_files = moving_val_files
        self.fixed_labels_files = fixed_labels


    def __len__(self):

        return len(self.fixed_files)

    def __getitem__(self, index):

        fixed_img = nib.load(self.fixed_files[index]).get_fdata()[np.newaxis, ...]
        moving_img = nib.load(self.moving_files[index]).get_fdata()[np.newaxis, ...]
        fixed_mask = nib.load(self.fixed_val_files[index]).get_fdata()[np.newaxis, ...]
        moving_mask = nib.load(self.moving_val_files[index]).get_fdata()[np.newaxis, ...]
        fixed_labels = nib.load(self.fixed_labels_files[index]).get_fdata()[np.newaxis, ...]
        return fixed_img, moving_img, fixed_mask, moving_mask, fixed_labels






from rigid_filed import get_reference_grid
from torch import nn
from typing import List, Optional, Sequence, Tuple, Union

class affine_to_dense(nn.Module):
    def __init__(
            self,
            image_size : Sequence[int] = (64, 224, 224),
            dtype=torch.float32,
            device='cuda:3'
    ) -> None:
        super().__init__()
        self._image_size = image_size
        self._dim = len(self._image_size)
        self._dtype = dtype
        self._device = device
        grid = get_reference_grid(self._image_size)
        grid = torch.cat([grid, torch.ones_like(grid[:1])]).to(self._device)
        self.register_buffer('grid', grid)


    def forward(self, transform_matrices, y_source_oh):
        rigid_flow = torch.einsum('qijk,bpq->bpijk', self.grid,
                                    transform_matrices.reshape(-1, 3, 4))
        rigid_flow = rigid_flow - self.grid[None, :3, ...]

        y_source_oh = y_source_oh.squeeze(0).unsqueeze(1)

        # (1,3,HWD), select displacement flow inside label areas
        rigid_flow = torch.sum(y_source_oh,
                                dim=0,
                                keepdim=True)*rigid_flow 
        
        return rigid_flow


import torch.nn.functional as F
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)

def pad_image(img, target_size):
    rows_to_pad = max(target_size[0] - img.shape[2], 0)
    cols_to_pad = max(target_size[1] - img.shape[3], 0)
    slcs_to_pad = max(target_size[2] - img.shape[4], 0)
    padded_img = F.pad(img, (0, slcs_to_pad, 0, cols_to_pad, 0, rows_to_pad), "constant", 0)
    return padded_img
