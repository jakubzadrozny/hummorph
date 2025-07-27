import torch.nn as nn
import torch.nn.functional as F

class BlendNet(nn.Module):
    def __init__(self, cfg):
        super(BlendNet, self).__init__()

    def forward(
        self,
        cnl_pts_norm,
        feat_volumes,
        **_,
    ):
        point_features = F.grid_sample(
            input=feat_volumes,
            grid=cnl_pts_norm[None, None, None, :, :],
            padding_mode="zeros",
            align_corners=True,
        )
        point_features = point_features[0, :, 0, 0, :]
        return point_features.T
