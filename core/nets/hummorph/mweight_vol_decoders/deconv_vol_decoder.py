import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils.network_util import ConvDecoder3D


class MotionWeightVolumeDecoder(nn.Module):
    def __init__(
            self,
            cfg,
            total_bones=24,
            num_subjects=1,
            pos_embed_fn=None, 
            pos_embed_size=None, 
            t_vertex=None, 
            in_vertex=False,
        ):
        super(MotionWeightVolumeDecoder, self).__init__()

        self.total_bones = total_bones
        self.volume_size = cfg.volume_size
        
        self.in_vertex = in_vertex

        self.embedding = nn.Embedding(
            num_embeddings=num_subjects,
            embedding_dim=cfg.embedding_size,
        )

        self.decoder = ConvDecoder3D(
            embedding_size=cfg.embedding_size,
            volume_size=cfg.volume_size, 
            voxel_channels=total_bones+1,
        )


    def forward(self,
                subject_idx,
                motion_weights_priors,
                **_):
        embeddings = self.embedding(subject_idx)

        decoded_weights = F.softmax(
            self.decoder(embeddings) + torch.log(motion_weights_priors), dim=1
        )
        return decoded_weights
