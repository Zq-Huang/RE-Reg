
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from rigid_filed import extract_class_masks


from kernels import (gradient_kernel_1d, gradient_kernel_2d,
                      gradient_kernel_3d, spatial_filter_nd)


def _grad_param(ndim, method, axis):
    if ndim == 1:
        kernel = gradient_kernel_1d(method)
    elif ndim == 2:
        kernel = gradient_kernel_2d(method, axis)
    elif ndim == 3:
        kernel = gradient_kernel_3d(method, axis)
    else:
        raise NotImplementedError

    kernel = kernel.reshape(1, 1, *kernel.shape)
    return Parameter(torch.tensor(kernel, dtype=torch.float32))




class IncompressibilityConstraint(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.grad_x_kernel = _grad_param(3, method='default', axis=0)
        self.grad_y_kernel = _grad_param(3, method='default', axis=1)
        self.grad_z_kernel = _grad_param(3, method='default', axis=2)
        self.grad_x_kernel.requires_grad = False
        self.grad_y_kernel.requires_grad = False
        self.grad_z_kernel.requires_grad = False

    def first_order_derivative(self, disp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            disp: the shape should be BCHWD, with B=1, C=3
        """
        # (3,1,1,H,W,D)
        gradx = torch.stack([
            0.5 * spatial_filter_nd(
                disp[:, [0], ...], kernel=self.grad_x_kernel, mode='constant'),
            0.5 * spatial_filter_nd(
                disp[:, [1], ...], kernel=self.grad_x_kernel, mode='constant'),
            0.5 * spatial_filter_nd(
                disp[:, [2], ...], kernel=self.grad_x_kernel, mode='constant')
        ],
                            dim=0)
        grady = torch.stack([
            0.5 * spatial_filter_nd(
                disp[:, [0], ...], kernel=self.grad_y_kernel, mode='constant'),
            0.5 * spatial_filter_nd(
                disp[:, [1], ...], kernel=self.grad_y_kernel, mode='constant'),
            0.5 * spatial_filter_nd(
                disp[:, [2], ...], kernel=self.grad_y_kernel, mode='constant')
        ],
                            dim=0)
        gradz = torch.stack([
            0.5 * spatial_filter_nd(
                disp[:, [0], ...], kernel=self.grad_z_kernel, mode='constant'),
            0.5 * spatial_filter_nd(
                disp[:, [1], ...], kernel=self.grad_z_kernel, mode='constant'),
            0.5 * spatial_filter_nd(
                disp[:, [2], ...], kernel=self.grad_z_kernel, mode='constant')
        ],
                            dim=0)

        # (3,3,1,H,W,D)
        grad_disp = torch.cat([gradx, grady, gradz], dim=1)
        # [dphi_x/dx, dphi_x/dy, dphi_x/dz]
        # [dphi_y/dx, dphi_y/dy, dphi_y/dz]
        # [dphi_z/dx, dphi_z/dy, dphi_z/dz]
        grad_deform = grad_disp + torch.eye(3, 3).view(3, 3, 1, 1, 1,
                                                       1).to(disp)

        return grad_deform

    def forward(self, y_source_oh: torch.Tensor,disp_field: torch.Tensor):
        """
        Compute the orthonormality condition of displacement field
        Args:
            y_source_oh (torch.tensor): (hard) one-hot format, the shape should be BNHW[D]
            disp_field (torch.tensor): the shape should be BCHWD, with B=1, C=3

        Returns:
            E_pc (torch.tensor): the properness condition energy
        """
        # exclude background
        y_source_oh = y_source_oh.squeeze(0).squeeze(0)

        # neg_y_source_oh = 1 - y_source_oh

        grad_deform = self.first_order_derivative(disp_field)

        # compute the Jacobian determinant
        pc = grad_deform[0, 0, ...] * (
                grad_deform[1, 1, ...] * grad_deform[2, 2, ...] - grad_deform[1, 2, ...] * grad_deform[2, 1, ...]) - \
             grad_deform[1, 0, ...] * (
                     grad_deform[0, 1, ...] * grad_deform[2, 2, ...] - grad_deform[0, 2, ...] * grad_deform[
                 2, 1, ...]) + \
             grad_deform[2, 0, ...] * (
                     grad_deform[0, 1, ...] * grad_deform[1, 2, ...] - grad_deform[0, 2, ...] * grad_deform[
                 1, 1, ...])

        pc = pc.squeeze()
        E_pc = 0

        y_source_oh_list = extract_class_masks(y_source_oh)
        for cls, _ in y_source_oh_list.items():
            y_mask_oh = y_source_oh_list[cls] 
            log_pc = torch.sum(torch.abs(torch.log(torch.clamp(abs(pc), min=1e-4))) * y_mask_oh)/ y_mask_oh.sum()
            E_pc += log_pc

        return E_pc / len(y_source_oh_list)

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        return repr_str






