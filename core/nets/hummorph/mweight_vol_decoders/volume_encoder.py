import torch
import torch.nn as nn
import torch.nn.functional as F

from core.nets.hummorph.component_factory import (
    load_volume_encoder,
    load_projector,
    load_mweight_vol_decoder,
)


class MotionWeightVolumeDecoder(nn.Module):
    def __init__(self, cfg, in_channels, total_bones=24, **_):
        super(MotionWeightVolumeDecoder, self).__init__()

        self.volume_size = cfg.volume_encoder.volume_size

        self.enable_in_iter = cfg.get("enable_in_iter", 0)
        self.use_initial_decoder = cfg.use_initial_decoder
        if self.use_initial_decoder:
            self.initial_decoder = load_mweight_vol_decoder(cfg.initial_decoder.module)(
                cfg=cfg.initial_decoder,
                total_bones=total_bones,
            )

        self.projector = load_projector(cfg.projector.module)()
        self.volume_encoder = load_volume_encoder(cfg.volume_encoder.module)(
            cfg=cfg.volume_encoder,
            in_channels=in_channels,
        )
        self.final_conv = nn.Conv3d(
            cfg.volume_encoder.model_channels, total_bones + 1, 1
        )

    def forward(
        self,
        subject_idx,
        src_imgs,
        featmaps,
        log_motion_weights_priors,
        in_motion_scale_Rs,
        in_motion_Ts,
        in_K,
        in_E,
        cnl_bbox_min_xyz,
        cnl_bbox_max_xyz,
        cnl_bbox_scale_xyz,
        iter,
        pos_enc_fn_xyz,
        pos_enc_fn_dir,
        canonical_joints,
    ):
        if self.use_initial_decoder:
            log_motion_weights_priors = self.initial_decoder(
                subject_idx,
                log_motion_weights_priors,
            ).squeeze(0)
        
        if iter < self.enable_in_iter:
            return log_motion_weights_priors
        
        motion_weights_priors = torch.exp(log_motion_weights_priors)
        feat_volumes = self.projector.get_undeformed_feature_volumes(
            volume_size=self.volume_size,
            src_imgs=src_imgs,
            featmaps=featmaps,
            motion_weights_vol=motion_weights_priors,
            in_motion_scale_Rs=in_motion_scale_Rs,
            in_motion_Ts=in_motion_Ts,
            in_K=in_K,
            in_E=in_E,
            cnl_bbox_min_xyz=cnl_bbox_min_xyz,
            cnl_bbox_max_xyz=cnl_bbox_max_xyz,
            cnl_bbox_scale_xyz=cnl_bbox_scale_xyz,
            pos_enc_fn_xyz=pos_enc_fn_xyz,
            pos_enc_fn_dir=pos_enc_fn_dir,
            canonical_joints=canonical_joints,
        )

        _, cnl_feat_volume = self.volume_encoder(feat_volumes.unsqueeze(0))
        mweights_corr = self.final_conv(cnl_feat_volume)
        if log_motion_weights_priors.shape[-1] == self.volume_size:
            log_motion_weights_priors = log_motion_weights_priors.unsqueeze(0)
        else:
            log_motion_weights_priors = F.interpolate(
                log_motion_weights_priors.unsqueeze(0),
                size=self.volume_size,
                mode='nearest',
                align_corners=True,
            )

        log_mweights_vol = F.log_softmax(
            mweights_corr + log_motion_weights_priors, dim=1
        )
        return log_mweights_vol
