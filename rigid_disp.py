import numpy as np
import small_gicp
import numpy
import torch
import torch.nn as nn
from typing import List,Sequence

def get_reference_grid(image_size: Sequence[int]) -> torch.Tensor:
    """
    Generate a unnormalized coordinate grid
    Args:
        image_size: shape of input image, e.g. (64,128,128)
    """
    mesh_points = [torch.arange(0, dim) for dim in image_size]
    grid = torch.stack(torch.meshgrid(*mesh_points),
                       dim=0).to(dtype=torch.float)  # (spatial_dims, ...)
    return grid


class rigid_to_dense(nn.Module):
    def __init__(
            self,
            image_size : Sequence[int] = (64, 224, 224),
            dtype=torch.float32,
            device='cpu'
    ) -> None:
        super().__init__()
        self._image_size = image_size
        self._dim = len(self._image_size)
        self._dtype = dtype
        self._device = device
        grid = get_reference_grid(self._image_size)
        grid = torch.cat([grid, torch.ones_like(grid[:1])]).to(self._device)
        self.register_buffer('grid', grid)


    def forward(self, transform_matrices, y_source_oh, return_orig = True):

        flow = torch.einsum('qijk,bpq->bpijk', self.grid,
                            transform_matrices.reshape(-1, 3, 4))
        if not return_orig:
            # normalize flow values to [-1, 1] for grid_sample
            for i in range(self._dim):
                flow[:, i, ...] = 2 * (flow[:, i, ...] /
                                       (self._image_size[i] - 1) - 0.5)

            # [X, Y, Z, [x,y,z]]
            flow = flow.permute([0] + list(range(2, 2 + self._dim)) + [1])
            index_ordering: List[int] = list(range(self._dim - 1, -1, -1))
            flow = flow[..., index_ordering]  # x,y,z -> z,y,x
        else:
            flow = flow - self.grid[None, :3, ...]
        y_source_oh = y_source_oh.squeeze(0).unsqueeze(1)
        rigid_flow = torch.sum(y_source_oh,
                        dim=0,
                        keepdim=True) * flow

        return rigid_flow
    

def mask_to_point_cloud(mask):
    """将掩膜转换为点云."""
    # 获取掩膜中非零体素的坐标
    points = np.array(np.where(mask > 0)).T  # 形状为 (N, 3)
    return points

def label_separate(mask):
    label_masks = []
    for label in range(1, int(mask.max()) + 1):
        label_sep = (mask == label)
        labels = mask * label_sep/ label
        label_masks.append(labels)
    return label_masks
   

  

def GICP(target_raw_numpy : numpy.ndarray, source_raw_numpy : numpy.ndarray):

  # Convert numpy arrays (Nx3 or Nx4) to small_gicp.PointCloud
  target_raw = small_gicp.PointCloud(target_raw_numpy)
  source_raw = small_gicp.PointCloud(source_raw_numpy)

  # Preprocess point clouds
  target, target_tree = small_gicp.preprocess_points(target_raw, downsampling_resolution=0.25)
  source, source_tree = small_gicp.preprocess_points(source_raw, downsampling_resolution=0.25)

  result = small_gicp.align(target, source, target_tree,  max_correspondence_distance = 10, num_threads = 12)

  return result.T_target_source


class rigid_flow():
    def __init__(
            self,
            image_size: Sequence[int] = (224, 224, 64),
            dtype=torch.float32,
            device='cpu') -> None:
        super().__init__()
        self._image_size = image_size
        self._dtype = dtype
        self._device = device
        self.rigid_to_dense = rigid_to_dense(image_size, device = self._device)

    def forward(self, sorce, target):
        sorce = sorce.squeeze(0).squeeze(0).numpy()
        target = target.squeeze(0).squeeze(0).numpy()
        mask_1 = label_separate(sorce)
        flow_f = []

        for i in range(min(4, len(mask_1))):
            mask1_i = mask_1[i]

            # 将掩膜转换为点云
            point_cloud1 = mask_to_point_cloud(mask1_i)
            point_cloud2 = mask_to_point_cloud(target)

            # ICP 配准
            transformation = GICP (
                target_raw_numpy=point_cloud2,  # 目标点云
                source_raw_numpy=point_cloud1,   # 源点云
            )
        
            trans = torch.from_numpy(transformation[:3, :]).unsqueeze(0).float()
            mask_i = torch.from_numpy(mask1_i).unsqueeze(0).unsqueeze(0).float()
            flow_i  = self.rigid_to_dense(trans, mask_i)
            flow_f.append(flow_i)
            
        flow = sum(flow_f)
        return flow

