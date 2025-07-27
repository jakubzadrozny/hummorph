from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import cfg
from core.nets.hummorph.deformation_helpers import _sample_motion_fields, _posed_view_dir_to_cnl
from core.nets.hummorph.component_factory import (
    load_feature_extractor,
    load_positional_embedder,
    load_canonical_mlp,
    load_mweight_vol_decoder,
    load_volume_encoder,
    load_global_feature_encoder,
    load_blend_net,
    load_projector,
)
from core.utils.network_util import MotionBasisComputer
from core.utils.body_util import closest_distance_to_points


class Network(nn.Module):
    def __init__(self, num_subjects=1):
        super(Network, self).__init__()

        # motion basis computer
        self.motion_basis_computer = MotionBasisComputer(total_bones=cfg.total_bones)
        
        self.feature_extractor = load_feature_extractor(cfg.feature_extractor.module)(
            coarse_only=True,
            coarse_out_ch=cfg.feature_extractor.out_dim,
        )

        # canonical positional encoding
        get_embedder = load_positional_embedder(cfg.embedder.module)
        self.cnl_pos_embed_fn, cnl_pos_embed_size = get_embedder(
            cfg.canonical_mlp.multires,
            cfg.canonical_mlp.i_embed,
        )
        self.cnl_pos_embed_fn_dir, cnl_pos_embed_dir_dim = get_embedder(
            multires=4, i=cfg.canonical_mlp.i_embed
        )
        
        self.feat_vol_pos_enc_xyz_fn, feat_vol_pos_enc_xyz_dim = get_embedder(
            cfg.feature_volume.embedder_xyz.multires,
            cfg.feature_volume.embedder_xyz.i_embed,
        )
        self.feat_vol_pos_enc_dir_fn, feat_vol_pos_enc_dir_dim = get_embedder(
            cfg.feature_volume.embedder_dir.multires,
            cfg.feature_volume.embedder_dir.i_embed,
        )

        # motion weight volume
        self.mweight_vol_decoder = load_mweight_vol_decoder(cfg.mweight_volume.module)(
            cfg.mweight_volume,
            total_bones=cfg.total_bones,
            num_subjects=num_subjects,
            featmaps_n_channels=3+self.feature_extractor.out_ch,
            pos_embed_xyz_dim=feat_vol_pos_enc_xyz_dim,
            pos_embed_dir_dim=feat_vol_pos_enc_dir_dim,
        )

        if cfg.global_feature.enabled:
            self.global_feature_encoder = load_global_feature_encoder(cfg.global_feature.module)(
                cfg.global_feature,
            )

        self.fg_thread = cfg.fg_thread
        self.projector = load_projector(cfg.projector.module)()
        if cfg.feature_volume.enabled:
            feat_vol_encoder_in_channels = (
                3 + self.feature_extractor.out_ch + 2*feat_vol_pos_enc_xyz_dim
                + feat_vol_pos_enc_dir_dim + 1 + self.mweight_vol_decoder.hidden_dim
            )
            self.volume_encoder = load_volume_encoder(cfg.feature_volume.volume_encoder.module)(
               cfg=cfg.feature_volume.volume_encoder,
               in_channels=feat_vol_encoder_in_channels,
            )
            feat_vol_n_channels = cfg.feature_volume.volume_encoder.model_channels
        else:
            feat_vol_n_channels = 0

        mweights_n_channels = self.mweight_vol_decoder.hidden_dim
        self.blend_net = load_blend_net(cfg.blend_net.module)(
            cfg=cfg.blend_net,
            featmaps_n_channels=3+self.feature_extractor.out_ch,
            feat_vol_n_channels=feat_vol_n_channels,
            mweights_n_channels=mweights_n_channels,
            pos_embed_xyz_dim=cnl_pos_embed_size,
            pos_embed_dir_dim=cnl_pos_embed_dir_dim,
        )

        # non-rigid motion st positional encoding
        # self.get_non_rigid_embedder = \
        #     load_positional_embedder(cfg.non_rigid_embedder.module)

        # non-rigid motion MLP
        # _, non_rigid_pos_embed_size = \
        #     self.get_non_rigid_embedder(cfg.non_rigid_motion_mlp.multires, 
        #                                 cfg.non_rigid_motion_mlp.i_embed)
        # self.forward_mlp = \
        #     load_non_rigid_motion_mlp(cfg.non_rigid_motion_mlp.module)(
        #         pos_embed_size=non_rigid_pos_embed_size,
        #         condition_code_size=cfg.non_rigid_motion_mlp.condition_code_size,
        #         mlp_width=cfg.non_rigid_motion_mlp.mlp_width,
        #         mlp_depth=cfg.non_rigid_motion_mlp.mlp_depth,
        #         skips=cfg.non_rigid_motion_mlp.skips,
        #         backward=False)
        # self.backward_mlp = \
        #     load_non_rigid_motion_mlp(cfg.non_rigid_motion_mlp.module)(
        #         pos_embed_size=non_rigid_pos_embed_size,
        #         condition_code_size=cfg.non_rigid_motion_mlp.condition_code_size,
        #         mlp_width=cfg.non_rigid_motion_mlp.mlp_width,
        #         mlp_depth=cfg.non_rigid_motion_mlp.mlp_depth,
        #         skips=cfg.non_rigid_motion_mlp.skips,
        #         backward=True)

        # nerf_input_ch = cnl_pos_embed_size * 2 + cfg.rgb_in_dim
        nerf_input_ch = 2*cnl_pos_embed_size + cnl_pos_embed_dir_dim + cfg.rgb_in_dim + mweights_n_channels + 1
        self.nerf_mlp = load_canonical_mlp(cfg.canonical_mlp.module)(
            input_ch=nerf_input_ch,
            output_ch=4, 
            mlp_depth=cfg.canonical_mlp.mlp_depth, 
            mlp_width=cfg.canonical_mlp.mlp_width,
            skips=cfg.canonical_mlp.skips,
        )


    # @staticmethod
    # def _expand_input(input_data, total_elem):
    #     assert input_data.shape[0] == 1
    #     input_size = input_data.shape[1]
    #     return input_data.expand((total_elem, input_size))


    def _batchify_rays(self, rays_flat, batchify=False, **kwargs):
        chunk = cfg.chunk if batchify else max(rays_flat.shape[0], 1)
        all_ret = defaultdict(list)
        for i in range(0, rays_flat.shape[0], chunk):
            ret = self._render_rays(rays_flat[i:i+chunk], **kwargs)
            for k in ret:
                all_ret[k].append(ret[k])

        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}

        if 'loss_consis' in all_ret:
            all_ret['loss_consis'] = torch.mean(all_ret['loss_consis'])
        return all_ret


    @staticmethod
    def _raw2outputs(raw, raw_mask, z_vals, rays_d, bgcolor=None):
        def _raw2alpha(raw, dists, act_fn=F.relu):
            return 1.0 - torch.exp(-act_fn(raw)*dists)

        dists = z_vals[...,1:] - z_vals[...,:-1]

        infinity_dists = torch.Tensor([1e10])
        infinity_dists = infinity_dists.expand(dists[...,:1].shape).to(dists)

        dists = torch.cat([dists, infinity_dists], dim=-1) 
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
        rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
        alpha = _raw2alpha(raw[...,3], dists)  # [N_rays, N_samples]
        alpha = alpha * raw_mask[:, :, 0]
        
        weights = alpha * torch.cumprod(
            torch.cat([
                torch.ones((alpha.shape[0], 1)).to(alpha), 
                1.-alpha + 1e-10,
            ], dim=-1),
        dim=-1)[:, :-1]
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        acc_map = torch.sum(weights, -1)
        rgb_map = rgb_map + (1.-acc_map[...,None]) * bgcolor[None, :]/255.
        return rgb_map, acc_map, weights, depth_map


    @staticmethod
    def _unpack_ray_batch(ray_batch):
        rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] 
        bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2]) 
        near, far = bounds[...,0], bounds[...,1] 
        return rays_o, rays_d, near, far


    @staticmethod
    def _get_samples_along_ray(N_rays, near, far):
        t_vals = torch.linspace(0., 1., steps=cfg.N_samples).to(near)
        z_vals = near * (1.-t_vals) + far * (t_vals)
        return z_vals.expand([N_rays, cfg.N_samples]) 


    @staticmethod
    def _stratified_sampling(z_vals):
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        
        t_rand = torch.rand(z_vals.shape).to(z_vals)
        z_vals = lower + (upper - lower) * t_rand

        return z_vals

    def _render_rays(
            self, 
            ray_batch, 
            motion_scale_Rs,
            motion_Ts,
            motion_weights_vol,
            mweights_feat_grid,
            cnl_bbox_min_xyz,
            cnl_bbox_scale_xyz,
            in_K,
            in_E,
            in_motion_scale_Rs,
            in_motion_Ts,
            projector,
            src_imgs,
            featmaps,
            canonical_joints,
            feat_volumes=None,
            global_feature_planes=None,
            bgcolor=None,
            iter_val=None,
            dst_posevec=None,
            **_):
        N_rays = ray_batch.shape[0]
        rays_o, rays_d, near, far = self._unpack_ray_batch(ray_batch)

        z_vals = self._get_samples_along_ray(N_rays, near, far)
        # print('perturb----', cfg.perturb, flush=True)
        if cfg.perturb > 0.:
            z_vals = self._stratified_sampling(z_vals)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
        depths = torch.sqrt(torch.sum((pts - rays_o[..., None, :])**2, dim=-1))
        ori_shape = pts.shape
        pts = pts.reshape(-1, 3)
        depths = depths.flatten()

        cnl_pts, pts_mask, backwarp_motion_weights = self.backward_deform(
            x_o=pts,
            motion_scale_Rs=motion_scale_Rs, 
            motion_Ts=motion_Ts, 
            motion_weights_vol=motion_weights_vol,
            cnl_bbox_min_xyz=cnl_bbox_min_xyz, 
            cnl_bbox_scale_xyz=cnl_bbox_scale_xyz,
        )
        fg_index = pts_mask > self.fg_thread
        fg_index = fg_index.squeeze(1)

        cnl_pts_norm = (cnl_pts - cnl_bbox_min_xyz[None, :]) * cnl_bbox_scale_xyz[None, :] - 1.0
        
        mweights = F.grid_sample(
            input=motion_weights_vol.unsqueeze(0),
            grid=cnl_pts_norm[None, None, None, :, :],
            padding_mode="zeros",
            align_corners=True,
        )
        mweights = mweights[0, :, 0, 0, :].T

        mweights_feats = F.grid_sample(
            input=mweights_feat_grid.unsqueeze(0),
            grid=cnl_pts_norm[None, None, None, :, :],
            padding_mode="zeros",
            align_corners=True,
        )
        mweights_feats = mweights_feats[0, :, 0, 0, :].T

        pts_o = _sample_motion_fields(
            pts=cnl_pts[fg_index],
            backwarp_motion_weights=mweights[fg_index],
            motion_scale_Rs=motion_scale_Rs[0], 
            motion_Ts=motion_Ts[0], 
            output_list=['x_skel'],
            backward=False,
        )['x_skel']

        distance = torch.norm(pts[fg_index] -  pts_o, dim=1)
        loss_consis = distance * (distance > 0.05).to(torch.float32)

        rgb_feat, _, _, feat_view_dir_cnl, feat_depth = projector.compute_for_canonical_pts(
            cnl_pts=cnl_pts,
            cnl_pts_mweights=mweights,
            src_imgs=src_imgs,
            featmaps=featmaps,
            in_K=in_K,
            in_E=in_E,
            in_motion_scale_Rs=in_motion_scale_Rs,
            in_motion_Ts=in_motion_Ts,
        )
        
        ray_dirs = rays_d / torch.sqrt(torch.sum(rays_d**2, dim=-1, keepdim=True))
        view_dirs = ray_dirs.unsqueeze(1).expand(-1, ori_shape[1], -1).reshape(-1, 3)
        cnl_view_dir = _posed_view_dir_to_cnl(
            view_dirs=view_dirs.unsqueeze(1),
            motion_scale_Rs=[motion_scale_Rs],
            backwarp_motion_weights=backwarp_motion_weights,
        ).squeeze(1)

        cnl_pts_dist, _ = closest_distance_to_points(canonical_joints, cnl_pts)
        query_pts_with_view_dir = torch.cat((
            self.cnl_pos_embed_fn(cnl_pts_norm),
            self.cnl_pos_embed_fn_dir(cnl_view_dir), # comment this to kill view dir
            self.cnl_pos_embed_fn(cnl_pts_dist),
            depths.unsqueeze(-1),
            mweights_feats,
        ), dim=-1)

        n_views = len(in_motion_scale_Rs)
        pts_with_view_dir = torch.cat((
            query_pts_with_view_dir.unsqueeze(1).expand(-1, n_views, -1),
            self.cnl_pos_embed_fn_dir(feat_view_dir_cnl),
            feat_depth.unsqueeze(-1),
        ), dim=-1)
        latent_feats = self.blend_net(
            cnl_pts_norm=cnl_pts_norm,
            query_pts_with_view_dir=query_pts_with_view_dir,
            rgb_feat=rgb_feat,
            pts_with_view_dir=pts_with_view_dir,
            feat_volumes=feat_volumes,
            global_feature_planes=global_feature_planes,
        )

        nerf_input = torch.cat((query_pts_with_view_dir, latent_feats), dim=-1)
        raws_flat = self.nerf_mlp(nerf_input)
        raws = raws_flat.reshape(list(ori_shape[:-1]) + [raws_flat.shape[-1]])

        pts_mask = pts_mask.reshape(list(ori_shape[:-1]) + [pts_mask.shape[-1]])
        rgb_map, acc_map, weights, depth_map = self._raw2outputs(
            raws, pts_mask, z_vals, rays_d, bgcolor
        )

        return {
            'rgb' : rgb_map,  
            'alpha' : acc_map, 
            'depth': depth_map,
            'weights': weights,
            'loss_consis':loss_consis,
        }


    def _get_motion_base(self, dst_Rs, dst_Ts, cnl_gtfms):
        motion_scale_Rs, motion_Ts = self.motion_basis_computer(
                                        dst_Rs, dst_Ts, cnl_gtfms)

        return motion_scale_Rs, motion_Ts


    @staticmethod
    def _multiply_corrected_Rs(Rs, correct_Rs):
        total_bones = cfg.total_bones - 1
        return torch.matmul(Rs.reshape(-1, 3, 3),
                            correct_Rs.reshape(-1, 3, 3)).reshape(-1, total_bones, 3, 3)
    

    def forward(self,
                rays,
                cnl_grid,
                dst_Rs, dst_Ts, cnl_gtfms,
                in_dst_Rs, in_dst_Ts,
                motion_weights_priors,
                src_imgs,
                in_K, 
                in_E,
                K, 
                E,
                cnl_bbox_min_xyz,
                cnl_bbox_scale_xyz,
                canonical_joints,
                mweights_dist_penalty,
                dst_posevec=None,
                near=None, far=None,
                iter_val=1e7,
                batchify=True,
                **kwargs):
        
       # torch.autograd.set_detect_anomaly(True)
        dst_Rs=dst_Rs[None, ...]
        dst_Ts=dst_Ts[None, ...]
        dst_posevec=dst_posevec[None, ...]
        cnl_gtfms=cnl_gtfms[None, ...]
        # motion_weights_priors=motion_weights_priors[None, ...]

        # non_rigid_pos_embed_fn, _ = \
        #     self.get_non_rigid_embedder(
        #         multires=cfg.non_rigid_motion_mlp.multires,                         
        #         is_identity=cfg.non_rigid_motion_mlp.i_embed,
        #         iter_val=iter_val,)

        featmaps, _ = self.feature_extractor(src_imgs.permute(0, 3, 1, 2))

        kwargs.update({
            # "non_rigid_pos_embed_fn": non_rigid_pos_embed_fn,
            "dst_posevec": dst_posevec,
            "iter_val":iter_val,
            "src_imgs":src_imgs,
            "featmaps": featmaps,
            'projector':self.projector,
            'in_K': in_K, 
            'in_E': in_E,
            'K':K, 
            'E':E,
        })

        r"""Compute motion bases between the target pose and canonical pose."""
        motion_scale_Rs, motion_Ts = self._get_motion_base(
            dst_Rs=dst_Rs,
            dst_Ts=dst_Ts,
            cnl_gtfms=cnl_gtfms,
        )
        in_motion_scale_Rs = []
        in_motion_Ts = []
        for dst_Rs_near, dst_Ts_near in zip(in_dst_Rs, in_dst_Ts):
            motion_scale_Rs_near, motion_Ts_near = self._get_motion_base(
                dst_Rs=dst_Rs_near[None, ...],
                dst_Ts=dst_Ts_near[None, ...],
                cnl_gtfms=cnl_gtfms,
            )
            in_motion_scale_Rs.append(motion_scale_Rs_near)
            in_motion_Ts.append(motion_Ts_near)

        grid_shape = cnl_grid.shape[:-1]
        cnl_grid_pts = cnl_grid.reshape(-1, 3)
        cnl_grid_pts_norm = (cnl_grid_pts - cnl_bbox_min_xyz[None, :]) * cnl_bbox_scale_xyz[None, :] - 1.0
        cnl_pts_dist, _ = closest_distance_to_points(canonical_joints, cnl_grid_pts)
        cnl_grid_spatial_feats = torch.cat((
            self.feat_vol_pos_enc_xyz_fn(cnl_grid_pts_norm),
            self.feat_vol_pos_enc_xyz_fn(cnl_pts_dist),
        ), dim=-1)
        cnl_grid_spatial_feats = cnl_grid_spatial_feats.reshape(*grid_shape, -1).permute(3, 0, 1, 2)

        # if iter_val >= cfg.enable_mweights_module:
        # log_motion_weights_priors = torch.nan_to_num(
        #     torch.log(motion_weights_priors), neginf=-1e6,
        # )
        log_motion_weights_priors = torch.log(motion_weights_priors)
        log_mweights_vol, cnl_grid_mweights_feats = self.mweight_vol_decoder(
            cnl_grid=cnl_grid,
            cnl_grid_pts_norm=cnl_grid_pts_norm,
            cnl_grid_spatial_feats=cnl_grid_spatial_feats,
            src_imgs=src_imgs,
            featmaps=featmaps,
            log_motion_weights_priors=log_motion_weights_priors,
            in_motion_scale_Rs=in_motion_scale_Rs,
            in_motion_Ts=in_motion_Ts,
            in_K=in_K,
            in_E=in_E,
            iter=iter_val,
            pos_enc_fn_dir=self.feat_vol_pos_enc_dir_fn,
        )
        motion_weights_vol = torch.exp(log_mweights_vol)

        cnl_grid_spatial_feats = torch.cat((
            cnl_grid_spatial_feats,
            cnl_grid_mweights_feats,
        ), dim=0)

        if cfg.feature_volume.enabled:
            _feat_volumes = self.projector.get_undeformed_feature_volumes(
                cnl_grid=cnl_grid,
                motion_weights_vol=motion_weights_vol,
                cnl_grid_spatial_feats=cnl_grid_spatial_feats,
                src_imgs=src_imgs,
                featmaps=featmaps,
                in_motion_scale_Rs=in_motion_scale_Rs,
                in_motion_Ts=in_motion_Ts,
                in_K=in_K,
                in_E=in_E,
                pos_enc_fn_dir=self.feat_vol_pos_enc_dir_fn,
            )
            _, feat_volumes, glob_latent = self.volume_encoder(_feat_volumes.unsqueeze(0))
            # _, feat_volumes = self.volume_encoder(_feat_volumes.unsqueeze(0))
        else:
            feat_volumes = None

        if cfg.global_feature.enabled:
            global_feature_planes = self.global_feature_encoder(glob_latent)
            #     global_feature_planes = self.global_feature_encoder(src_imgs.permute(0, 3, 1, 2))
            global_feature_planes = global_feature_planes.view(3, 32, global_feature_planes.shape[-2], global_feature_planes.shape[-1]) # [3, 32, 256, 256]
        else:
            global_feature_planes = None

        kwargs.update({
            'motion_scale_Rs': motion_scale_Rs,
            'motion_Ts': motion_Ts,
            'in_motion_scale_Rs': in_motion_scale_Rs,
            'in_motion_Ts': in_motion_Ts,
            'motion_weights_vol': motion_weights_vol,
            'mweights_feat_grid': cnl_grid_mweights_feats,
            'feat_volumes': feat_volumes,
            'global_feature_planes': global_feature_planes,
            'cnl_bbox_min_xyz': cnl_bbox_min_xyz,
            'cnl_bbox_scale_xyz': cnl_bbox_scale_xyz,
            'canonical_joints': canonical_joints,
        })

        rays_o, rays_d = rays
        rays_o = torch.reshape(rays_o, [-1,3]).float()
        rays_d = torch.reshape(rays_d, [-1,3]).float()
        packed_ray_infos = torch.cat([rays_o, rays_d, near, far], -1)

        all_ret = self._batchify_rays(packed_ray_infos, batchify=batchify, **kwargs)
        all_ret['log_mweights_vol'] = log_mweights_vol
        return all_ret

    def backward_deform(self, x_o, motion_scale_Rs, motion_Ts, motion_weights_vol, cnl_bbox_min_xyz, cnl_bbox_scale_xyz):
        '''
        Use sample weight and non-rigid mlp to deform points from observation space to canonical space.
        params: x_o, points in canonical space
        params: dst_posevec, pose parameters
        params: motion_scale_Rs, skel transfrom matrix from observation space to canonical space. Size: 24, 3, 3
        params: motion_Ts, translation vectors from observation space to canonical space. Size: 24, 3
        params: cnl_bbox_min_xyz, minxyz coords in canonical space.
        params: cnl_bbox_scale_xyz, scale factor for xyz coords in canonical space.
        '''

        mv_output = _sample_motion_fields(
            pts=x_o,
            motion_scale_Rs=motion_scale_Rs[0],
            motion_Ts=motion_Ts[0],
            motion_weights_vol=motion_weights_vol,
            cnl_bbox_min_xyz=cnl_bbox_min_xyz,
            cnl_bbox_scale_xyz=cnl_bbox_scale_xyz,
            output_list=['x_skel', 'fg_likelihood_mask', 'motion_weights'],
            backward=True,
        )

        pts_mask = mv_output['fg_likelihood_mask']
        x_c = mv_output['x_skel']
        motion_weights = mv_output['motion_weights']

        # non_rigid_embed_xyz = non_rigid_pos_embed_fn(x_c)
        # result = self.backward_mlp(
        #     pos_embed=non_rigid_embed_xyz,
        #     pos_xyz=x_c,
        #     condition_code=self._expand_input(dst_posevec, x_c.shape[0])
        # )
        # x_c = result['xyz']

        return x_c, pts_mask, motion_weights
