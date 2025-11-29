import torch
import torch.nn.functional as F
from torch import nn
from rigid_utils import sample_displacement_flow
    
import torch.nn.modules as nn

def sample_correspondence(label_map: torch.Tensor, flow: torch.Tensor,  intervals: float) -> torch.Tensor:
    """
    Sample correspondence between fixed and moving images
    Args:
        label_map: (soft) one-hot label mask of fixed image, tensor of shape BNHWD, with B=1
        flow: dense displacement field mapping from fixed image to moving image
        num_samples: number of sampled correspondences

    Returns:
        two corresponding point clouds
        src_pnts_list:
        dst_pnts_list:
    """
    image_size = label_map.shape[2:]
    num_ch = label_map.shape[1]
    src_pnts_list = []
    des_pnts_list = []

    # Define the real-world distance per pixel
    real_distance = torch.tensor([1.0, 1.0, 1.5], device=label_map.device)
    for ch in range(num_ch):
        # 获取所有有效点
        valid_points = (label_map[0, ch] >= 0.5).nonzero(as_tuple=True)
        valid_points = torch.stack(valid_points, dim=1).float()  # 转换为浮点数

        valid_points = valid_points[::intervals]  # 选择间隔为20的点
        # 计算流动
        sample_flow = sample_displacement_flow(valid_points, flow, image_size)
        sample_flow = sample_flow.squeeze()

        # Convert pixel coordinates to real-world coordinates
        src_pnts = valid_points.transpose(0, 1) * real_distance.unsqueeze(1)
        des_pnts = src_pnts + sample_flow * real_distance.unsqueeze(1)

        src_pnts_list.append(src_pnts)
        des_pnts_list.append(des_pnts)

    return src_pnts_list, des_pnts_list

def distances_loss(label_map: torch.Tensor, flow: torch.Tensor, intervals: float) -> torch.Tensor:
    """
    Compute distances between sampled source points and their corresponding transformed destination points.

    Args:
        label_map: (soft) one-hot label mask of fixed image, tensor of shape BNHWD, with B=1.
        flow: dense displacement field mapping from fixed image to moving image.
        num_samples: number of sampled correspondences.

    Returns:
        distances: Tensor of distances between consecutive source points and transformed destination points.
    """
    src_pnts_list, des_pnts_list= sample_correspondence(label_map, flow, intervals)
    src_pnts_list = src_pnts_list[0]
    des_pnts_list = des_pnts_list[0]
    distances = 0
    for i in range(1, len(src_pnts_list[0])):
        # 计算前一个点与后一个点的距离

        src_distance = torch.norm(src_pnts_list[:,i] - src_pnts_list[:,i - 1])
        des_distance = torch.norm(des_pnts_list[:,i] - des_pnts_list[:,i - 1])
        
        # 计算差值
        distance_diff = torch.abs(des_distance - src_distance)
        distances += distance_diff     

    return distances/len(src_pnts_list[0])

    
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


class SpatialTransformNearest_unit(nn.Module):
    def __init__(self):
        super(SpatialTransformNearest_unit, self).__init__()

    def forward(self, x, flow, sample_grid):

        sample_grid = sample_grid + flow
        flow = torch.nn.functional.grid_sample(x, sample_grid, mode='nearest', padding_mode="border", align_corners=True)
        return flow







