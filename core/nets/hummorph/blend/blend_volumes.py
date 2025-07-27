import torch
import torch.nn as nn
import torch.nn.functional as F

class BlendNet(nn.Module):
    def __init__(self, cfg):
        super(BlendNet, self).__init__()
        self.cfg = cfg
        self.rgb_fc = nn.Sequential(
            nn.Linear(35, 36),
            nn.ReLU(),
            nn.Linear(36, 36),
            nn.ReLU(),
        )
        self.pts_fc = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.weight_fc = nn.Sequential(
            nn.Linear(36+32, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.volume_weight_fc = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.pix_linear = nn.Linear(35, 32)
        self.volume_linear = nn.Linear(32, 32)

    def forward(
        self,
        cnl_pts_norm,
        feat_volumes,
        rgb_feat,
        pts_with_view_dir,
        **_,
    ):
        point_feat = F.grid_sample(
            input=feat_volumes,
            grid=cnl_pts_norm[None, None, None, :, :],
            padding_mode="zeros",
            align_corners=True,
        )
        point_feat = point_feat[0, :, 0, 0, :].T
        volume_weight = self.volume_weight_fc(point_feat)

        # global_feat = torch.cat([rgb_feat, torch.zeros(list(rgb_feat.shape[:-1]) + [1]).to(rgb_feat)], dim=-1)
        x = torch.cat((
            self.rgb_fc(rgb_feat),
            self.pts_fc(pts_with_view_dir),
        ), dim=-1)
        pix_weight = self.weight_fc(x)
        
        pix_feat = self.pix_linear(rgb_feat)
        volume_feat = self.volume_linear(point_feat)
        all_feats = torch.cat(
            (pix_feat, volume_feat.unsqueeze(-2)), dim=-2
        )
        all_weights = torch.cat(
            (pix_weight, volume_weight.unsqueeze(-2)), dim=-2
        )
        all_weights = F.softmax(all_weights, dim=-2)
        final_feat = torch.sum(all_weights * all_feats, dim=-2)
        return final_feat
