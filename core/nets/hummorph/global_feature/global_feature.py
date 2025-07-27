# S-Lab License 1.0

# Copyright 2023 S-Lab

# Redistribution and use for non-commercial purpose in source and binary forms, 
# with or without modification, are permitted provided that the following conditions are met: 
# 1. Redistributions of source code must retain the above copyright notice, 
# this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, 
# this list of conditions and the following disclaimer in the documentation and/or 
# other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors 
# may be used to endorse or promote products derived from this software 
# without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY 
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES 
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
# SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, 
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT 
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# 4. In the event that redistribution and/or use for commercial purpose in source or binary forms, 
# with or without modification is required, please contact the contributor(s) of the work.


import torch
import torch.nn as nn
from torchvision.models import resnet18

from core.nets.hummorph.global_feature.stylegan2 import Generator as StyleGAN2Backbone


class ResNet18Classifier(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ResNet18Classifier, self).__init__()
        self.backbone = resnet18(pretrained=True)

    def forward(self, x, extract_feature=False):
        # x = self.backbone(x)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        if not extract_feature:
            x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        if extract_feature:
            return x
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)

        return x


class GlobalFeatureEncoder(nn.Module):
    def __init__(self, cfg, **_):
        super(GlobalFeatureEncoder, self).__init__()

        synthesis_kwargs = {
            'channel_base': 32768,
            'channel_max': 512,
            'fused_modconv_default': 'inference_only',
            'num_fp16_res': 0,
            'conv_clamp': None,
        }
        mapping_kwargs = {
            'num_layers': cfg.map_depth,
        }
        self.backbone = StyleGAN2Backbone(
            z_dim=512,
            c_dim=0,
            w_dim=512,
            img_resolution=256,
            img_channels=32*3,
            mapping_kwargs=mapping_kwargs,
            **synthesis_kwargs,
        )


    def mapping(self, glob_latent, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        z = glob_latent
        c = torch.zeros((1, 25)).to(z.device)
        return self.backbone.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)


    def forward(self, glob_latent):
        ws = self.mapping(glob_latent)
        planes = self.backbone.synthesis(ws, update_emas=False, noise_mode='none')
        return planes[0]
