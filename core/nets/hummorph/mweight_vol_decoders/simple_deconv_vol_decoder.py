import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils.network_util import ConvDecoder3D


class MotionWeightVolumeDecoder(nn.Module):
    def __init__(
            self,
            cfg,
            total_bones=24,
            in_vertex=False,
            **_,
        ):
        super(MotionWeightVolumeDecoder, self).__init__()

        self.total_bones = total_bones
        self.volume_size = cfg.volume_size
        
        self.in_vertex = in_vertex

        self.const_embedding = nn.Parameter(
            torch.randn(cfg.embedding_size), requires_grad=True 
        )

        self.decoder = ConvDecoder3D(
            embedding_size=cfg.embedding_size,
            volume_size=cfg.volume_size, 
            voxel_channels=total_bones+1,
        )


    def forward(self,
                log_motion_weights_priors,
                **_):
        embedding = self.const_embedding[None, ...]
        mweight_corr = self.decoder(embedding)
        log_decoded_weights = F.log_softmax(
            mweight_corr + log_motion_weights_priors, dim=1
        )
        return log_decoded_weights
