import torch
import torch.nn.functional as F

from ..deformation_helpers import (
    _posed_view_dir_to_cnl,
    _sample_motion_fields,
)


class Projector:
    """
    find correspondences in correspondences bank
    """

    def __init__(self) -> None:
        pass

    def compute_projections(self, pts, in_K, in_E):
        """
        project 3d points to pixels
        :param pts: N, 3
        :param in_K: 2, 3, 3
        :param in_E: 2, 4, 4
        :return location: 2, N, 2; mask: 2, N
        """
        Rt = in_E
        tvec = Rt[:, :3, 3]
        R = Rt[:, :3, :3]
        cam_o = -torch.matmul(R.transpose(1, 2), tvec.unsqueeze(-1)).squeeze(-1)
        ray_d = pts - cam_o.unsqueeze(1)
        depth = torch.sqrt(torch.sum(ray_d**2, dim=-1, keepdim=True))
        ray_d = ray_d / depth

        v_cam = torch.einsum("...ik,...kj->...ij", pts, R.transpose(1, 2))
        v_cam = v_cam + tvec[:, None, :]
        v_project = torch.einsum("...ik,...kj->...ij", v_cam, in_K.transpose(1, 2))
        v_pixel = v_project[:, :, :2] / torch.clamp(v_project[:, :, 2:3], min=1e-8)
        v_pixel = torch.clamp(v_pixel, min=-1e6, max=1e6)
        mask = v_project[..., 2] > 0

        return v_pixel, mask, ray_d, depth.squeeze(-1)

    def normalize(self, pixel_locations, h, w):
        resize_factor = torch.tensor([w - 1.0, h - 1.0]).to(pixel_locations.device)[
            None, None, :
        ]
        normalized_pixel_locations = (
            2 * pixel_locations / resize_factor - 1.0
        )  # [n_views, n_points, 2]
        return normalized_pixel_locations

    def inbound(self, pixel_locations, h, w):
        """
        check if the pixel locations are in valid range
        :param pixel_locations: [..., 2]
        :param h: height
        :param w: weight
        :return: mask, bool, [...]
        """
        return (
            (pixel_locations[..., 0] <= w - 1.0)
            & (pixel_locations[..., 0] >= 0)
            & (pixel_locations[..., 1] <= h - 1.0)
            & (pixel_locations[..., 1] >= 0)
        )

    def compute(self, pts, in_K, in_E, train_imgs, featmaps):
        """
        :param pts,: [N, 3]
        :param in_K: 2, 3, 3
        :param in_E: 2, 4, 4
        :param train_imgs: [1, 2, h, w, 3]
        :param featmaps: [n_views, d, h, w]
        :return: rgb_feat_sampled: [n_rays, n_samples, 3+n_feat],train_imgs
        """

        train_imgs = train_imgs.permute(0, 3, 1, 2)  # [n_views, 3, h, w]

        h, w = train_imgs[0].shape[-2:]
        # compute the projection of the query points to each reference image
        pixel_locations, mask_in_front, ray_d, depth = self.compute_projections(
            pts, in_K, in_E
        )
        # pixel_locations: M, N, 2

        normalized_pixel_locations = self.normalize(
            pixel_locations, h, w
        )  # [n_views, n_pts, 2]

        # rgb sampling
        rgb_sampled = (
            F.grid_sample(
                train_imgs,
                normalized_pixel_locations.unsqueeze(2),
                align_corners=True,
            )
            .squeeze(3)
            .permute(2, 0, 1)
        )  # [n_pts, n_views, 3]

        # deep feature sampling
        feat_sampled = (
            F.grid_sample(
                featmaps,
                normalized_pixel_locations.unsqueeze(2),
                align_corners=True,
            )
            .squeeze(3)
            .permute(2, 0, 1)
        )  # [n_pts, n_views, d]
        rgb_feat_sampled = torch.cat(
            [rgb_sampled, feat_sampled], dim=-1
        )  # [n_pts, n_views, d+3]

        # mask, view direction, depth
        inbound = self.inbound(pixel_locations, h, w)
        mask = (inbound * mask_in_front).float().T  # [n_pts, n_views]
        ray_d = ray_d.permute(1, 0, 2)
        depth = depth.T
        return rgb_feat_sampled, mask, pixel_locations, ray_d, depth

    def compute_for_canonical_pts(
        self,
        cnl_pts,
        cnl_pts_mweights,
        src_imgs,
        featmaps,
        in_K,
        in_E,
        in_motion_scale_Rs,
        in_motion_Ts,
    ):
        motion_out = _sample_motion_fields(
            pts=cnl_pts,
            backwarp_motion_weights=cnl_pts_mweights,
            output_list=["x_skel", "motion_weights"],
            motion_scale_Rs=in_motion_scale_Rs[0][0],
            motion_Ts=in_motion_Ts[0][0],
            backward=False,
        )
        pts_ref = [motion_out["x_skel"].reshape(-1, 3)]
        backwarp_motion_weights = motion_out["motion_weights"].reshape(
            pts_ref[0].shape[0], -1
        )

        n_views = len(in_motion_scale_Rs)
        for view_idx in range(1, n_views):
            motion_out = _sample_motion_fields(
                pts=cnl_pts,
                backwarp_motion_weights=cnl_pts_mweights,
                output_list=["x_skel"],
                motion_scale_Rs=in_motion_scale_Rs[view_idx][0],
                motion_Ts=in_motion_Ts[view_idx][0],
                backward=False,
            )
            pts_ref.append(motion_out["x_skel"].reshape(-1, 3))

        pts_ref = torch.stack(pts_ref, dim=0)
        rgb_feat_sampled, mask, pixel_locations, feat_view_dir, depth = self.compute(
            pts_ref, in_K, in_E, src_imgs, featmaps
        )

        feat_view_dir_cnl = _posed_view_dir_to_cnl(
            view_dirs=feat_view_dir,
            motion_scale_Rs=in_motion_scale_Rs,
            backwarp_motion_weights=backwarp_motion_weights,
        )

        return rgb_feat_sampled, mask, pixel_locations, feat_view_dir_cnl, depth


    def get_undeformed_feature_volumes(
        self,
        cnl_grid,
        motion_weights_vol,
        cnl_grid_spatial_feats,
        src_imgs,
        featmaps,
        in_motion_scale_Rs,
        in_motion_Ts,
        in_K,
        in_E,
        pos_enc_fn_dir,
    ):
        vol_coords_shape = cnl_grid.shape[:-1]
        cnl_vol_coords = cnl_grid.reshape(-1, 3)

        cnl_pts_mweights = motion_weights_vol.reshape(-1, cnl_vol_coords.shape[0]).T
        feats, _, _, feat_view_dir_cnl, feat_depth = self.compute_for_canonical_pts(
            cnl_pts=cnl_vol_coords,
            cnl_pts_mweights=cnl_pts_mweights,
            src_imgs=src_imgs,
            featmaps=featmaps,
            in_K=in_K,
            in_E=in_E,
            in_motion_scale_Rs=in_motion_scale_Rs,
            in_motion_Ts=in_motion_Ts,
        )
        feat_view_dir_pos_enc = pos_enc_fn_dir(feat_view_dir_cnl)
        n_views = len(in_motion_scale_Rs)
        feats = torch.cat((
            feats,
            feat_view_dir_pos_enc,
            feat_depth.unsqueeze(2),
        ), dim=2)
        
        feat_volumes = (
            feats.permute(1, 0, 2)
            .reshape(n_views, *vol_coords_shape, -1)
            .permute(0, 4, 1, 2, 3)
        )
        feat_volumes = torch.cat((
            feat_volumes,
            cnl_grid_spatial_feats.expand(n_views, -1, -1, -1, -1),
        ), dim=1)
        return feat_volumes
