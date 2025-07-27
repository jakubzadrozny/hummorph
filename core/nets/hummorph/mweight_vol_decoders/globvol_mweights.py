import torch
import torch.nn as nn
import torch.nn.functional as F

from core.nets.hummorph.component_factory import (
    load_volume_encoder,
    load_projector,
    load_global_feature_encoder,
)
from core.utils.network_util import ConvDecoder3D, initmod, project_onto_planes


def sample_from_planes(plane_features, cnl_pts_norm):
    projected_coordinates = project_onto_planes(cnl_pts_norm) # [3, 1, 786432, 2]
    output_features = F.grid_sample(
        input=plane_features,
        grid=projected_coordinates.unsqueeze(1),
        padding_mode='zeros',
        align_corners=True,
    )
    # .permute(0, 3, 2, 1).reshape(n_planes, M, C)
    return torch.mean(output_features.squeeze(2), dim=0)


class MotionWeightVolumeDecoder(nn.Module):
    def __init__(self, cfg, featmaps_n_channels, pos_embed_xyz_dim, pos_embed_dir_dim, total_bones=24, **_):
        super(MotionWeightVolumeDecoder, self).__init__()

        self.volume_size = cfg.volume_encoder.volume_size
        self.hidden_dim = cfg.feature_dim

        self.enable_in_iter = cfg.get("enable_in_iter", 0)
        
        self.const_embedding = nn.Parameter(
            torch.randn(cfg.embedding_size), requires_grad=True 
        )

        self.initial_decoder = ConvDecoder3D(
            embedding_size=cfg.embedding_size,
            volume_size=cfg.volume_size, 
            voxel_channels=cfg.feature_dim,
        )

        self.init_conv = nn.Conv3d(cfg.feature_dim, total_bones+1, 1)
        initmod(self.init_conv)

        self.init_linear = nn.Conv3d(cfg.feature_dim, cfg.feature_dim, 1)
        self.vol_linear = nn.Conv3d(cfg.feature_dim, cfg.feature_dim, 1)
        self.glob_linear = nn.Conv3d(cfg.feature_dim, cfg.feature_dim, 1)
        self.final_linear = nn.Conv3d(cfg.feature_dim, total_bones+1, 1)
        initmod(self.init_linear)
        initmod(self.vol_linear)
        initmod(self.glob_linear)
        initmod(self.final_linear)

        self.projector = load_projector(cfg.projector.module)()
        self.volume_encoder = load_volume_encoder(cfg.volume_encoder.module)(
            cfg=cfg.volume_encoder,
            in_channels=(
                featmaps_n_channels + 2*pos_embed_xyz_dim + pos_embed_dir_dim + cfg.feature_dim + 1
            ),
        )
        # self.volume_encoder.apply(initmod)

        self.global_feature_encoder = load_global_feature_encoder(cfg.global_feature.module)(
            cfg.global_feature,
        )
        # self.global_feature_encoder.apply(initmod)


    def forward(
        self,
        cnl_grid,
        cnl_grid_pts_norm,
        cnl_grid_spatial_feats,
        src_imgs,
        featmaps,
        log_motion_weights_priors,
        in_motion_scale_Rs,
        in_motion_Ts,
        in_K,
        in_E,
        iter,
        pos_enc_fn_dir,
    ):
        grid_shape = cnl_grid.shape[:-1]
        
        embedding = self.const_embedding[None, ...]
        initial_feats = self.initial_decoder(embedding)

        initial_mweights_corr = self.init_conv(initial_feats)
        log_initial_mweights_vol = F.log_softmax(
            initial_mweights_corr + log_motion_weights_priors, dim=1
        ).squeeze(0)
        
        if iter < self.enable_in_iter:
            return log_initial_mweights_vol, initial_feats.squeeze(0)

        initial_mweights_vol = torch.exp(log_initial_mweights_vol)
        _grid_spatial_feats = torch.cat((
            cnl_grid_spatial_feats.unsqueeze(0),
            initial_feats,
        ), dim=1)

        feat_volumes = self.projector.get_undeformed_feature_volumes(
            cnl_grid=cnl_grid,
            motion_weights_vol=initial_mweights_vol,
            cnl_grid_spatial_feats=_grid_spatial_feats,
            src_imgs=src_imgs,
            featmaps=featmaps,
            in_motion_scale_Rs=in_motion_scale_Rs,
            in_motion_Ts=in_motion_Ts,
            in_K=in_K,
            in_E=in_E,
            pos_enc_fn_dir=pos_enc_fn_dir,
        )

        _, cnl_feat_volume, glob_latent = self.volume_encoder(feat_volumes.unsqueeze(0))

        global_feature_planes = self.global_feature_encoder(glob_latent)
        global_feature_planes = global_feature_planes.view(3, 32, global_feature_planes.shape[-2], global_feature_planes.shape[-1]) # [3, 32, 256, 256]
        grid_global_feat = sample_from_planes(global_feature_planes, cnl_grid_pts_norm)
        
        grid_global_feat = grid_global_feat.reshape(32, *grid_shape)

        initial_feats = self.init_linear(initial_feats)
        vol_feats = self.vol_linear(cnl_feat_volume)
        global_feats = self.glob_linear(grid_global_feat.unsqueeze(0))
        combined_feats = initial_feats + vol_feats + global_feats

        final_mweight_corr = self.final_linear(combined_feats)

        log_mweights_vol = F.log_softmax(
            final_mweight_corr + log_motion_weights_priors, dim=1
        ).squeeze(0)
        return log_mweights_vol, combined_feats.squeeze(0)
