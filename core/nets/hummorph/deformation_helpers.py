import torch
from torch.nn import functional as F

from configs import cfg


def _sample_motion_fields(
    pts,
    output_list,
    motion_weights_vol=None,
    cnl_bbox_min_xyz=None,
    cnl_bbox_scale_xyz=None,
    motion_scale_Rs=None,
    motion_Ts=None,
    backward=True,
    backwarp_motion_weights=None,
):
    if backward:
        # remove BG channel
        motion_weights = motion_weights_vol[:-1]

        ###from observation space to canonical space
        pos = torch.matmul(
            motion_scale_Rs.unsqueeze(1),
            pts.unsqueeze(0).unsqueeze(-1),
        ).squeeze(-1) + motion_Ts.unsqueeze(1)

        pos_normalized = (pos - cnl_bbox_min_xyz[None, None, :]) * cnl_bbox_scale_xyz[None, None, :] - 1.0
        weights = F.grid_sample(
            input=motion_weights.unsqueeze(1),
            grid=pos_normalized[:, None, None, ...],
            padding_mode="zeros",
            align_corners=True,
        )
        backwarp_motion_weights = weights[:, 0, 0, 0, :].T
    else:
        if backwarp_motion_weights is None:
            raise ValueError('mweights cannot be none when backward is false')
        backwarp_motion_weights = backwarp_motion_weights[:, :-1]

    backwarp_motion_weights_sum = torch.sum(
        backwarp_motion_weights, dim=-1, keepdim=True
    )

    if not backward:
        inv_R = motion_scale_Rs.transpose(1, 2)        
        pos = torch.matmul(
            inv_R.unsqueeze(1), 
            (pts.unsqueeze(0) - motion_Ts.unsqueeze(1)).unsqueeze(-1),
        ).squeeze(-1)
        
    normalized_motion_weights = backwarp_motion_weights / backwarp_motion_weights_sum.clamp(min=0.0001)
    x_skel = torch.sum(
        normalized_motion_weights.T.unsqueeze(-1) * pos, dim=0,
    )
    fg_likelihood_mask = (
        torch.clamp(backwarp_motion_weights_sum, min=0, max=1)
        if cfg.clamp_fg_likelihood
        else backwarp_motion_weights_sum
    )

    results = {}
    if "x_skel" in output_list:  # [N_rays x N_samples, 3]
        results["x_skel"] = x_skel
    if "fg_likelihood_mask" in output_list:  # [N_rays x N_samples, 1]
        results["fg_likelihood_mask"] = fg_likelihood_mask
    if "motion_weights" in output_list:
        results["motion_weights"] = normalized_motion_weights

    return results


def _posed_view_dir_to_cnl(
    view_dirs,
    motion_scale_Rs,
    backwarp_motion_weights,
):
    in_Rs = torch.cat(motion_scale_Rs, dim=0)
    # view_dirs is [n_pts, n_views, 3]
    # backwarp_motion_weights is [n_pts, n_bones]
    # in_Rs is [n_views, n_bones, 3, 3]
    # in_weighted_Rs is [n_pts, n_views, 3, 3]
    _in_weighted_Rs = torch.sum(
        backwarp_motion_weights[:, None, :, None, None] * in_Rs[None, ...], dim=2
    )  # .reshape(*ori_shape[:-1], n_views, 3, 3)
    backwarp_motion_weights_sum = torch.sum(backwarp_motion_weights, dim=-1)
    in_weighted_Rs = (
        _in_weighted_Rs
        + (1 - backwarp_motion_weights_sum)[:, None, None, None]
        * torch.eye(3, device=_in_weighted_Rs.device)[None, None, ...]
    )

    view_dir_cnl = torch.matmul(
        in_weighted_Rs, view_dirs.unsqueeze(-1)
    ).squeeze(-1)  # [n_pts, n_views, 3]
    # can this (below) explode? (if norm is close to zero)
    view_dir_cnl_norm = torch.sqrt(
        torch.sum(view_dir_cnl**2, dim=-1, keepdim=True)
    )
    view_dir_cnl = view_dir_cnl / view_dir_cnl_norm.clamp(min=0.0001)
    return view_dir_cnl
