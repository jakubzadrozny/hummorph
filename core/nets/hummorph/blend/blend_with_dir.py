# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import cfg

# default tensorflow initialization of linear layers
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


class BlendNet(nn.Module):
    def __init__(
            self,
            cfg,
            pos_embed_xyz_dim,
            pos_embed_dir_dim,
            featmaps_n_channels,
            hidden_dim=64,
            **_,
        ):
        super(BlendNet, self).__init__()
        self.cfg = cfg
        activation_func = nn.ReLU(inplace=False)
        self.rgb_fc = nn.Sequential(
            nn.Linear(featmaps_n_channels+1, hidden_dim),
            activation_func,
            nn.Linear(hidden_dim, hidden_dim),
            activation_func,
        )
        self.pts_fc = nn.Sequential(
            nn.Linear(2*pos_embed_xyz_dim+2*pos_embed_dir_dim+2, hidden_dim),
            activation_func,
            nn.Linear(hidden_dim, hidden_dim),
            activation_func,
        )
        self.weight_fc = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            activation_func,
            nn.Linear(hidden_dim, 1),
        )
        self.rgb_fc.apply(weights_init)
        self.pts_fc.apply(weights_init)
        self.weight_fc.apply(weights_init)


    def forward(self, rgb_feat, pts_with_view_dir, **_):
        '''
        params: @rgb_feat: rgbs and image features [n_rays, n_samples, n_views, n_feat]
        return: blend rgb_feat [n_rays, n_samples, 35]
        '''
        
        global_feat = torch.cat([rgb_feat, torch.zeros(list(rgb_feat.shape[:-1]) + [1]).to(rgb_feat)], dim=-1)
        x = torch.cat((
            self.rgb_fc(global_feat),
            self.pts_fc(pts_with_view_dir),
        ), dim=-1)
        blend_weight = self.weight_fc(x)
        blend_weight = F.softmax(blend_weight, dim=-2)
        rgb_feat = torch.sum(rgb_feat * blend_weight, dim=-2)

        return rgb_feat

