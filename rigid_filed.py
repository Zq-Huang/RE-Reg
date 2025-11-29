from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn


from rigid_utils import (get_mass_center, get_reference_grid,
                          sample_correspondence, sample_displacement_flow,
                          solve_SVD)


class RigidFieldLoss(nn.Module):
    """Compute rigid motion between input one-hot mask and target one-hot mask using
    SVD.

    Args:
            image_size: shape of input image
            num_samples: int, number of sampling correspondences
            downsize: int, downsampling factor for computer field MSE loss
            inv: bool, flag deciding the direction of sampling correspondences
    """
    # 导入参数
    def __init__(
            self,
            image_size: Sequence[int] = (64, 224, 224),
            num_samples: int = 256,
            # downsize: int = 2,
            inv: bool = False,
            include_background: bool = True,
            dtype=torch.float32,
            device='cuda') -> None:
        super().__init__()
        self._image_size = image_size
        self._dim = len(self._image_size)
        self._dtype = dtype
        self._device = device
        grid = get_reference_grid(self._image_size)
        grid = torch.cat([grid, torch.ones_like(grid[:1])]).to(self._device)
        # (4,HWD)
        self.register_buffer('grid', grid)
        self.num_samples = num_samples
        self.inv = inv
        self.include_background = include_background
        # self.resize = ResizeFlow(scale_factor=downsize, ndim=self._dim)

    # 实现了论文"使用SVD的最小二乘刚性运动"中描述的最小二乘法求解刚性运动的方法。
    # 输入为两个点集，两个质心
    def lsq_rigid_motion(self, y_source_pnts_list: Sequence[torch.Tensor],
                         source_pnts_list: Sequence[torch.Tensor],
                         y_source_cm_list: torch.Tensor,
                         source_cm_list: torch.Tensor) -> torch.Tensor:
        """
        Least Square Method solving Rigid motion from correspondences
        https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
        Args:
            y_source_pnts_list: list of length N, tensor of shape (3, num_samples)
            source_pnts_list: list of length N, tensor of shape (3, num_samples)
            y_source_cm_list: center mass of y_source_oh, tensor of shape (3, N)
            source_cm_list: center mass of source_oh, tensor of shape (3, N)

        Returns:
            transform_matrix: tensor of shape (N, 4, 4)
        """
        num_ch = len(y_source_pnts_list)
        trans_matrix_list = []
        for ch in range(num_ch):
            # points in fixed image 取点集中的第ch个点
            y_source_pnts = y_source_pnts_list[ch]
            # corresponding points in target image
            source_pnts = source_pnts_list[ch]
            # 计算两个点集的加权质心
            y_source_cm = y_source_cm_list[:, [ch]]
            source_cm = source_cm_list[:, [ch]]
            # 用奇异值算旋转和平移矩阵
            R, t = solve_SVD(y_source_pnts, source_pnts, y_source_cm,
                             source_cm)
            # 将旋转和平移矩阵构建为一个4×4的矩阵
            trans_matrix_pos = torch.diag(
                torch.ones(4, dtype=self._dtype, device=self._device))
            trans_matrix_rot = torch.diag(
                torch.ones(4, dtype=self._dtype, device=self._device))

            trans_matrix_pos[:3, [3]] = t
            trans_matrix_rot[:3, :3] = R

            trans_matrix = trans_matrix_pos @ trans_matrix_rot
            trans_matrix_list.append(trans_matrix)

        # (N, 4, 4)，返回刚性变换矩阵
        transform_matrices = torch.stack(trans_matrix_list, dim=0)
        return transform_matrices[:, :3, :]

    def forward(self,
                y_source_oh: torch.Tensor,
                source_oh: torch.Tensor,
                flow: torch.Tensor,
                neg_flow: Optional[torch.Tensor] = None) -> torch.Tensor:
        """

        Args:
            y_source_oh (torch.tensor): one-hot mask, tensor of shape BNHWD, with B=1
            source_oh (torch.tensor): one-hot mask, tensor of shape BNHWD, with B=1
            flow (torch.tensor): tensor of shape B3HWD, with B=1, dense displacement field mapping from y_source_oh to source_oh
            neg_flow (torch.tensor | None): tensor of shape B3HWD, with B=1, dense displacement field mapping from source_oh to y_source_oh

        Returns:

        """


        with torch.no_grad():
            if not self.include_background:  # 看输入张量的维度是否包含背景通道
                y_source_oh = y_source_oh[:, 1:]
                source_oh = source_oh[:, 1:]

            # BNHWD  即5个维度
            # exclude low volume mask
            # valid_ch = torch.logical_and(
            #     y_source_oh.sum(dim=(0, 2, 3, 4)) > 0,
            #     source_oh.sum(dim=(0, 2, 3, 4)) > 0)
            # y_source_oh = y_source_oh[:, valid_ch, ...]
            # source_oh = source_oh[:, valid_ch, ...]

            # 求解质心
            y_source_cm_list = get_mass_center(y_source_oh, self.grid,
                                               self._dim)
            source_cm_list = get_mass_center(source_oh, self.grid, self._dim)

            if self.inv and neg_flow is not None:
                source_pnts_list, y_source_pnts_list = sample_correspondence(
                    source_oh, neg_flow, self.num_samples)
            else:
                y_source_pnts_list, source_pnts_list = sample_correspondence(
                    y_source_oh, flow, self.num_samples)


            transform_matrices = self.lsq_rigid_motion(y_source_pnts_list,
                                                       source_pnts_list,
                                                       y_source_cm_list,
                                                       source_cm_list)

            # (N1HWD)
            y_source_oh = y_source_oh.squeeze(0).unsqueeze(1)

            # (N3HWD), N=self._num_ch, [[x,y,z], X,Y,Z]
            rigid_flow = torch.einsum('qijk,bpq->bpijk', self.grid,
                                      transform_matrices.reshape(-1, 3, 4))
            rigid_flow = rigid_flow - self.grid[None, :3, ...]

            # (1,3,HWD), select displacement flow inside label areas
            rigid_flow = torch.sum( y_source_oh,
                                   dim=0,
                                   keepdim=True)*rigid_flow 

            # compute the downsized rigid flow
            # rigid_flow = self.resize(rigid_flow)

        # (1,3,HWD)，将刚性变换转化为流场
        flow = torch.sum(y_source_oh, dim=0, keepdim=True) * flow

        # the output displacement flow of voxelmorph network is actually an upsampled flow
        # the final output of the registration head is downsized flow
        # flow = self.resize(flow)

        # loss = torch.mean(torch.linalg.norm(rigid_flow - flow, dim=1))
        loss = torch.linalg.norm(rigid_flow - flow,
                                 dim=1).sum() / y_source_oh.sum()  # 计算了二范数
        return loss
    
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += (f'(image_size={self._image_size},'
                     f'num_samples={self.num_samples},'
                     f'inv={self.inv},'
                     f'include_background={self.include_background},'
                     f'dtype={self._dtype},'
                     f'device={self._device})')
        return repr_str
    
#--------------------------------------------------多类别的刚性损失函数--------------------------------------------------------------

def extract_class_masks(mask):

    # 获取所有唯一标签，并按标签排序
    unique_classes = sorted(mask.unique().tolist())  # 将标签转换为列表并排序
    unique_classes = [cls for cls in unique_classes if 1 <= cls < 5] 
    extracted_masks = {}

    for cls in unique_classes:
        # 创建一个与原 mask 相同形状的全零张量
        cls_mask = torch.zeros_like(mask, dtype=torch.float)  # 保持mask的类型
        # 提取对应类别的标签
        cls_mask[mask == cls] = 1  # 标记为1的部分是该类别
        extracted_masks[int(cls)] = cls_mask  # 将类别和对应的 mask 存储在字典中

    return extracted_masks



class  MultiRigidFieldLoss(nn.Module):
    """Compute rigid motion between input one-hot mask and target one-hot mask using
    SVD.

    Args:
            image_size: shape of input image
            num_samples: int, number of sampling correspondences
            downsize: int, downsampling factor for computer field MSE loss
            inv: bool, flag deciding the direction of sampling correspondences
    """
    # 导入参数
    def __init__(
            self,
            image_size: Sequence[int] = (64, 224, 224),
            num_samples: int = 256,
            # downsize: int = 2,
            inv: bool = False,
            include_background: bool = True,
            dtype=torch.float32,
            device='cuda') -> None:
        super().__init__()
        self._image_size = image_size
        self._dim = len(self._image_size)
        self._dtype = dtype
        self._device = device
        grid = get_reference_grid(self._image_size)
        grid = torch.cat([grid, torch.ones_like(grid[:1])]).to(self._device)
        # (4,HWD)
        self.register_buffer('grid', grid)
        self.num_samples = num_samples
        self.inv = inv
        self.include_background = include_background
        # self.resize = ResizeFlow(scale_factor=downsize, ndim=self._dim)

    # 实现了论文"使用SVD的最小二乘刚性运动"中描述的最小二乘法求解刚性运动的方法。
    # 输入为两个点集，两个质心
    def lsq_rigid_motion(self, y_source_pnts_list: Sequence[torch.Tensor],
                         source_pnts_list: Sequence[torch.Tensor],
                         y_source_cm_list: torch.Tensor,
                         source_cm_list: torch.Tensor) -> torch.Tensor:
        """
        Least Square Method solving Rigid motion from correspondences
        https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
        Args:
            y_source_pnts_list: list of length N, tensor of shape (3, num_samples)
            source_pnts_list: list of length N, tensor of shape (3, num_samples)
            y_source_cm_list: center mass of y_source_oh, tensor of shape (3, N)
            source_cm_list: center mass of source_oh, tensor of shape (3, N)

        Returns:
            transform_matrix: tensor of shape (N, 4, 4)
        """
        num_ch = len(y_source_pnts_list)
        trans_matrix_list = []
        for ch in range(num_ch):
            # points in fixed image 取点集中的第ch个点
            y_source_pnts = y_source_pnts_list[ch]
            # corresponding points in target image
            source_pnts = source_pnts_list[ch]
            # 计算两个点集的加权质心
            y_source_cm = y_source_cm_list[:, [ch]]
            source_cm = source_cm_list[:, [ch]]
            # 用奇异值算旋转和平移矩阵
            R, t = solve_SVD(y_source_pnts, source_pnts, y_source_cm,
                             source_cm)
            # 将旋转和平移矩阵构建为一个4×4的矩阵
            trans_matrix_pos = torch.diag(
                torch.ones(4, dtype=self._dtype, device=self._device))
            trans_matrix_rot = torch.diag(
                torch.ones(4, dtype=self._dtype, device=self._device))

            trans_matrix_pos[:3, [3]] = t
            trans_matrix_rot[:3, :3] = R

            trans_matrix = trans_matrix_pos @ trans_matrix_rot
            trans_matrix_list.append(trans_matrix)

        # (N, 4, 4)，返回刚性变换矩阵
        transform_matrices = torch.stack(trans_matrix_list, dim=0)
        return transform_matrices[:, :3, :]

    def forward(self,
                y_source_oh: torch.Tensor,
                source_oh: torch.Tensor,
                flow: torch.Tensor,
                neg_flow: Optional[torch.Tensor] = None) -> torch.Tensor:
        """

        Args:
            y_source_oh (torch.tensor): one-hot mask, tensor of shape BNHWD, with B=1
            source_oh (torch.tensor): one-hot mask, tensor of shape BNHWD, with B=1
            flow (torch.tensor): tensor of shape B3HWD, with B=1, dense displacement field mapping from y_source_oh to source_oh
            neg_flow (torch.tensor | None): tensor of shape B3HWD, with B=1, dense displacement field mapping from source_oh to y_source_oh
            flow 表示从moving到fixed
        Returns:

        """

        with torch.no_grad():
            if not self.include_background:  # 看输入张量的维度是否包含背景通道
                y_source_oh = y_source_oh[:, 1:]
                source_oh = source_oh[:, 1:]

            # BNHWD  即5个维度
            # exclude low volume mask
            # valid_ch = torch.logical_and(
            #     y_source_oh.sum(dim=(0, 2, 3, 4)) > 0,
            #     source_oh.sum(dim=(0, 2, 3, 4)) > 0)
            # y_source_oh = y_source_oh[:, valid_ch, ...]
            # source_oh = source_oh[:, valid_ch, ...]

            # 求解质心
            y_source_cm_list = get_mass_center(y_source_oh, self.grid,
                                            self._dim)
            source_cm_list = get_mass_center(source_oh, self.grid, self._dim)

            if self.inv and neg_flow is not None:
                source_pnts_list, y_source_pnts_list = sample_correspondence(
                    source_oh, neg_flow, self.num_samples)
            else:
                y_source_pnts_list, source_pnts_list = sample_correspondence(
                    y_source_oh, flow, self.num_samples)


            transform_matrices = self.lsq_rigid_motion(y_source_pnts_list,
                                                    source_pnts_list,
                                                    y_source_cm_list,
                                                    source_cm_list)
            


            # (N1HWD)
            y_source_oh = y_source_oh.squeeze(0).unsqueeze(1)

            # (N3HWD), N=self._num_ch, [[x,y,z], X,Y,Z]
            rigid_flow = torch.einsum('qijk,bpq->bpijk', self.grid,
                                    transform_matrices.reshape(-1, 3, 4))
            rigid_flow = rigid_flow - self.grid[None, :3, ...]

            # (1,3,HWD), select displacement flow inside label areas
            rigid_flow = torch.sum( y_source_oh,
                                dim=0,
                                keepdim=True)*rigid_flow 
            
            

            

            # compute the downsized rigid flow
            # rigid_flow = self.resize(rigid_flow)

        # (1,3,HWD)，将刚性变换转化为流场
        flow = torch.sum(y_source_oh, dim=0, keepdim=True) * flow

        # the output displacement flow of voxelmorph network is actually an upsampled flow
        # the final output of the registration head is downsized flow
        # flow = self.resize(flow)

        # loss = torch.mean(torch.linalg.norm(rigid_flow - flow, dim=1))
        loss = torch.linalg.norm(rigid_flow - flow,
                                dim=1).sum() / y_source_oh.sum()  # 计算了二范数

        return loss


    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += (f'(image_size={self._image_size},'
                     f'num_samples={self.num_samples},'
                     f'inv={self.inv},'
                     f'include_background={self.include_background},'
                     f'dtype={self._dtype},'
                     f'device={self._device})')
        return repr_str


def save_mask_as_nifti(mask, filename, affine):
    # 创建 NIfTI 图像对象
    nifti_img = nib.Nifti1Image(mask, affine = affine)
    
    # 保存为 .nii.gz 文件
    nib.save(nifti_img, filename)

class SpatialTransformNearest_unit(nn.Module):
    def __init__(self):
        super(SpatialTransformNearest_unit, self).__init__()

    def forward(self, x, flow, sample_grid):

        sample_grid = sample_grid + flow
        flow = torch.nn.functional.grid_sample(x, sample_grid, mode='nearest', padding_mode="border", align_corners=True)
        return flow


import nibabel as nib 
from Functions import generate_grid_unit

import numpy as np

if __name__ == "__main__":

    imgshape = (224, 224, 64)
    grid = generate_grid_unit(imgshape)
    grid = torch.from_numpy(np.reshape(grid, (1,) + grid.shape))

    transform_label = SpatialTransformNearest_unit()

    rigid_field_loss = MultiRigidFieldLoss(
        image_size=(224, 224, 64),
        num_samples=256,
        inv=False,
        include_background=True,
        dtype=torch.float32,
        device='cpu'  # 或'cpu'
    )

    mask = torch.from_numpy(nib.load("/home/zhiqih/project/data/Task2/val_pl/2PA005/ct_bone-pro.nii.gz").get_fdata())
    affine = nib.load("/home/zhiqih/project/data/Task2/val_pl/2PA005/ct_bone-pro.nii.gz").affine
    mask = mask.unsqueeze(0).unsqueeze(0)
    flow = torch.from_numpy(nib.load("/home/zhiqih/project/data/Task2/val_pl/2PA005/LAPIRN_rig_Loss_flow.nii.gz").get_fdata())
    flow = flow.unsqueeze(0).float()
    mask_warp = transform_label(mask, flow.permute(0, 2, 3, 4, 1), grid)

    rigid_penalty, rigid_flow = rigid_field_loss(mask, mask_warp, -1 * flow)
    save_mask_as_nifti(rigid_flow.permute(0,2,3,4,1).numpy()[0,:,:,:,:], "/home/zhiqih/project/data/flow.nii.gz", affine)

  



