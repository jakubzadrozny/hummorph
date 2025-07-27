import numpy as np

from configs import cfg


class BadMaskException(Exception):
    pass


def select_rays(select_inds, rays_o, rays_d, ray_img, near, far):
    rays_o = rays_o[select_inds]
    rays_d = rays_d[select_inds]
    ray_img = ray_img[select_inds]
    near = near[select_inds]
    far = far[select_inds]
    return rays_o, rays_d, ray_img, near, far


def _get_patch_ray_indices(
    ray_mask, 
    candidate_mask, 
    patch_size, 
    H, W, bbox_bounds,
):
    assert len(ray_mask.shape) == 1
    assert ray_mask.dtype == bool
    assert candidate_mask.dtype == bool

    valid_ys, valid_xs = np.where(candidate_mask)
    if valid_ys.size == 0:
        raise BadMaskException

    half_patch_size = patch_size // 2
    # TODO: enable below to have more even sampling?
    # x_patch_in_bbox = ((valid_xs - half_patch_size) >= bbox_bounds["xmin"]) & ((valid_xs + half_patch_size) <= bbox_bounds["xmax"])
    # y_patch_in_bbox = ((valid_ys - half_patch_size) >= bbox_bounds["ymin"]) & ((valid_ys + half_patch_size) <= bbox_bounds["ymax"])
    # is_valid_center = x_patch_in_bbox & y_patch_in_bbox
    # valid_ys = valid_ys[is_valid_center]
    # valid_xs = valid_xs[is_valid_center]

    # determine patch center
    select_idx = np.random.choice(valid_ys.shape[0], size=[1], replace=False)[0]
    center_x = valid_xs[select_idx]
    center_y = valid_ys[select_idx]

    # determine patch boundary
    x_min = np.clip(a=center_x-half_patch_size, 
                    a_min=bbox_bounds["xmin"], 
                    # a_min=0,
                    a_max=bbox_bounds["xmax"]-patch_size)
                    # a_max=W-patch_size)
    x_max = x_min + patch_size
    y_min = np.clip(a=center_y-half_patch_size,
                    a_min=bbox_bounds["ymin"],
                    # a_min=0,
                    a_max=bbox_bounds["ymax"]-patch_size)
                    # a_max=H-patch_size)
    y_max = y_min + patch_size

    sel_ray_mask = np.zeros_like(candidate_mask)
    sel_ray_mask[y_min:y_max, x_min:x_max] = True

    #####################################################
    ## Below we determine the selected ray indices
    ## and patch valid mask

    sel_ray_mask = sel_ray_mask.reshape(-1)
    inter_mask = np.bitwise_and(sel_ray_mask, ray_mask)
    select_masked_inds = np.where(inter_mask)

    masked_indices = np.cumsum(ray_mask) - 1
    select_inds = masked_indices[select_masked_inds]
    
    inter_mask = inter_mask.reshape(H, W)

    # if select_inds.shape[0] < 0.5*patch_size*patch_size:
        # warnings.warn(f"small patch sampled ({select_inds.shape[0]} valid rays, {patch_size**2} expected)")
        # print("CENTER", center_x, center_y)
        # print("BOUNDS", x_min, x_max, y_min, y_max)
        # print("BBOX BOUNDS", bbox_bounds)
        # raise RuntimeError(f"{select_inds.shape[0]} < {0.7*patch_size*patch_size}")

    return select_inds, \
            inter_mask[y_min:y_max, x_min:x_max], \
            np.array([x_min, y_min]), np.array([x_max, y_max])
    

def get_patch_ray_indices(
    N_patch, 
    ray_mask, 
    subject_mask, 
    bbox_mask,
    patch_size, 
    H, W
):
    assert subject_mask.dtype == bool
    assert bbox_mask.dtype == bool

    bbox_exclude_subject_mask = np.bitwise_and(
        bbox_mask,
        np.bitwise_not(subject_mask)
    )

    valid_ys, valid_xs = np.where(bbox_mask)
    bbox_bounds = dict(
        xmin=np.min(valid_xs),
        xmax=np.max(valid_xs),
        ymin=np.min(valid_ys),
        ymax=np.max(valid_ys),
    )

    list_ray_indices = []
    list_mask = []
    list_xy_min = []
    list_xy_max = []

    total_rays = 0
    patch_div_indices = [total_rays]
    for _ in range(N_patch):
        # let p = cfg.patch.sample_subject_ratio
        # prob p: we sample on subject area
        # prob (1-p): we sample on non-subject area but still in bbox
        if np.random.rand(1)[0] < cfg.patch.sample_subject_ratio:
            candidate_mask = subject_mask
        else:
            candidate_mask = bbox_exclude_subject_mask

        ray_indices, mask, xy_min, xy_max = _get_patch_ray_indices(
            ray_mask, candidate_mask, patch_size, H, W, bbox_bounds,
        )

        assert len(ray_indices.shape) == 1
        total_rays += len(ray_indices)

        list_ray_indices.append(ray_indices)
        list_mask.append(mask)
        list_xy_min.append(xy_min)
        list_xy_max.append(xy_max)
        
        patch_div_indices.append(total_rays)

    select_inds = np.concatenate(list_ray_indices, axis=0)
    patch_info = {
        'mask': np.stack(list_mask, axis=0),
        'xy_min': np.stack(list_xy_min, axis=0),
        'xy_max': np.stack(list_xy_max, axis=0)
    }
    patch_div_indices = np.array(patch_div_indices)

    return select_inds, patch_info, patch_div_indices


def sample_patch_rays(img, H, W,
    subject_mask, bbox_mask, ray_mask,
    rays_o, rays_d, ray_img, near, far):

    select_inds, patch_info, patch_div_indices = get_patch_ray_indices(
        N_patch=cfg.patch.N_patches, 
        ray_mask=ray_mask, 
        subject_mask=subject_mask, 
        bbox_mask=bbox_mask,
        patch_size=cfg.patch.size, 
        H=H, 
        W=W,
    )

    rays_o, rays_d, ray_img, near, far = select_rays(
        select_inds, rays_o, rays_d, ray_img, near, far
    )
        
    targets = []
    for i in range(cfg.patch.N_patches):
        x_min, y_min = patch_info['xy_min'][i] 
        x_max, y_max = patch_info['xy_max'][i]
        targets.append(img[y_min:y_max, x_min:x_max])
    target_patches = np.stack(targets, axis=0) # (N_patches, P, P, 3)

    patch_masks = patch_info['mask']  # boolean array (N_patches, P, P)

    return rays_o, rays_d, ray_img, near, far, \
            target_patches, patch_masks, patch_div_indices, patch_info
