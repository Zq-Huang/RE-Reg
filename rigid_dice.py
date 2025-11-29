from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from monai.losses import DiceLoss
from monai.utils.enums import LossReduction
from torch.nn.modules.loss import _Loss
from torch import nn

from rigid_utils import RigidTransformation, get_closest_rigid

from rigid_utils import (get_mass_center, get_reference_grid,
                          sample_correspondence, sample_displacement_flow,
                          solve_SVD)

class RigidDiceLoss(_Loss):
    """Compute the dice loss between the prediction and closest rigidly transformed
    label.

    Args:
            include_background (bool): whether to include the background channel in Dice loss computation.
            reduction (LossReduction, str): {``"none"``, ``"mean"``, ``"sum"``}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.

                - ``"none"``: no reduction will be applied.
                - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
                - ``"sum"``: the output will be summed.
    """

    # TODO: add device to __init__
    def __init__(
            self,
            include_background: bool = True,
            reduction: Union[LossReduction, str] = LossReduction.MEAN) -> None:
        super().__init__(reduction=LossReduction(reduction).value)
        self.include_background = include_background
        # the first channel won't be background, inputs are one-hot format
        self.dice_loss_func = DiceLoss(include_background=True,
                                       to_onehot_y=False,
                                       reduction=self.reduction)

    def forward(self,
                y_source_oh: torch.Tensor,
                source_oh: torch.Tensor,
                flow: torch.Tensor,
                neg_flow: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            source_oh (torch.tensor): (hard) one-hot format, the shape should be BNHW[D], B=1
            y_source_oh (torch.tensor): (soft) one-hot format, the shape should be BNHW[D], B=1
            flow (torch.tensor): displacement field, tensor of shape (13HWD)

        Return:
            dice_loss (torch.tensor): the dice loss between the prediction and closest rigidly transformed label.
        """
        if not self.include_background:
            source_oh = source_oh[:, 1:]
            y_source_oh = y_source_oh[:, 1:]

        # BNHWD
        # exclude low volume mask
        # valid_ch = torch.logical_and(
        #     y_source_oh.sum(dim=(0, 2, 3, 4)) > 100,
        #     source_oh.sum(dim=(0, 2, 3, 4)) > 100)
        # y_source_oh = y_source_oh[:, valid_ch, ...]
        # source_oh = source_oh[:, valid_ch, ...]

        # rigid_y_source_mask is soft one-hot
        rigid_y_source_mask, rigid_flow = get_closest_rigid(
            source_oh.detach(), y_source_oh.detach(), flow.detach())
        dice_loss = self.dice_loss_func(y_source_oh, rigid_y_source_mask)
        return dice_loss


    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += (f'(include_background={self.include_background},'
                     f'reduction={self.reduction})')
        return repr_str
    



class RigidDiceLoss2(_Loss):

    # TODO: add device to __init__
    def __init__(
            self,
            image_size: Sequence[int] = (224, 224, 64),
            include_background: bool = True,
            reduction: Union[LossReduction, str] = LossReduction.MEAN,
            num_samples: int = 256,
            inv: bool = False,
            dtype=torch.float32,
            device='cuda') -> None:
        super().__init__(reduction=LossReduction(reduction).value)
        self.include_background = include_background
        # the first channel won't be background, inputs are one-hot format
        self.dice_loss_func = DiceLoss(include_background=True,
                                       to_onehot_y=False,
                                       reduction=self.reduction)
        
        self._device = device
        self._image_size = image_size
        self._dim = len(self._image_size)
        self._dtype = dtype
        grid = get_reference_grid(self._image_size)
        grid = torch.cat([grid, torch.ones_like(grid[:1])]).to(self._device)
        # (4,HWD)
        self.register_buffer('grid', grid)
        self.num_samples = num_samples
        self.inv = inv
        
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
            source_oh (torch.tensor): (hard) one-hot format, the shape should be BNHW[D], B=1
            y_source_oh (torch.tensor): (soft) one-hot format, the shape should be BNHW[D], B=1
            flow (torch.tensor): displacement field, tensor of shape (13HWD)

        Return:
            dice_loss (torch.tensor): the dice loss between the prediction and closest rigidly transformed label.
        """
        with torch.no_grad():
            if not self.include_background:
                source_oh = source_oh[:, 1:]
                y_source_oh = y_source_oh[:, 1:]

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

            trans = RigidTransformation(source_oh,
                                                opt_cm=False,
                                                dtype=self._dtype,
                                                device=self._device)

            flow = trans._compute_dense_flow(transform_matrices, return_orig=False)

            rigid_y_source_mask = F.grid_sample(source_oh, grid = flow, mode='nearest', align_corners=True)
            
            dice_loss = self.dice_loss_func(y_source_oh, rigid_y_source_mask)

            return dice_loss


    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += (f'(include_background={self.include_background},'
                     f'reduction={self.reduction})')
        return repr_str
    
class DiceLoss2(nn.Module):
    def __init__(self):
        super(DiceLoss2, self).__init__()
        self.eps = 1e-5

    def forward(self, input, target):
        N = target.size(0)

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2. * (intersection.sum(1) + self.eps) / (input_flat.sum(1) + target_flat.sum(1) + self.eps)
        loss = 1. - loss.sum() / N

        return loss

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor).cuda()

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)






    









