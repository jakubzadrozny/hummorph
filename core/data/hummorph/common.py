import pickle

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from trimesh import Trimesh

from configs import cfg
from core.utils.image_util import load_image


def skeleton_to_bbox(skeleton, offset=None):
    if offset is None:
        offset = cfg.bbox_offset

    min_xyz = np.min(skeleton, axis=0) - offset
    max_xyz = np.max(skeleton, axis=0) + offset
    return {
        'min_xyz': min_xyz,
        'max_xyz': max_xyz
    }


def vertices_to_bbox(vertices):
    # min_xyz = np.min(vertices, axis=0)
    # max_xyz = np.max(vertices, axis=0)
    # bbox_size = (max_xyz - min_xyz) * (1 + cfg.bbox_margin)
    # bbox_center = (min_xyz + max_xyz) / 2
    # min_xyz = bbox_center - bbox_size / 2
    # max_xyz = bbox_center + bbox_size / 2
    min_xyz = np.min(vertices, axis=0) - cfg.canonical_bbox_offset
    max_xyz = np.max(vertices, axis=0) + cfg.canonical_bbox_offset
    return {
        'min_xyz': min_xyz,
        'max_xyz': max_xyz
    }


def load_image_with_mask(subject_dir, frame_name, camera, extension="jpg", bgcolor=None, scale=None):
    img_path = subject_dir / "images" / f"{frame_name}.{extension}"
    mask_path = subject_dir / "masks" / f"{frame_name}.png"
    orig_img = np.array(load_image(str(img_path)))
    alpha_mask = np.array(load_image(str(mask_path)))

    # undistort image
    if 'distortions' in camera:
        K = camera['intrinsics']
        D = camera['distortions']
        orig_img = cv2.undistort(orig_img, K, D)
        alpha_mask = cv2.undistort(alpha_mask, K, D)

    if bgcolor is None:
        bgcolor = np.zeros(3)
    alpha_mask = alpha_mask / 255.
    img = alpha_mask * orig_img + (1.0 - alpha_mask) * bgcolor[None, None, :]

    if scale is None:
        scale = cfg.resize_img_scale
    if scale != 1.:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        alpha_mask = cv2.resize(alpha_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        
    img = (img / 255.).astype('float32')  
    return img, alpha_mask


def load_canonical_joints(subject_dir, shift_pelvis=True):
    cl_joint_path = subject_dir / "canonical_joints.pkl"
    with cl_joint_path.open("rb") as f:
        cl_joint_data = pickle.load(f)
    canonical_joints = cl_joint_data['joints']

    if isinstance(canonical_joints, dict):
        canonical_joints = canonical_joints['joints']

    canonical_joints = canonical_joints.astype('float32')
    if shift_pelvis:
        pelvis_pos = canonical_joints[0].copy()
        canonical_joints -= pelvis_pos
    canonical_bbox = skeleton_to_bbox(canonical_joints)
    return canonical_joints, canonical_bbox


def get_canonical_bbox(smpl_model, dst_skel):
    smpl_scale = dst_skel["smpl_scale"]

    vertices, _, canonical_joints = smpl_model(np.zeros_like(dst_skel["poses"]), dst_skel["avg_betas"])
    canonical_joints *= smpl_scale
    vertices *= smpl_scale
    pelvis_pos = np.copy(canonical_joints[0])
    vertices -= pelvis_pos
    
    return vertices_to_bbox(vertices)


def get_estimated_canonical_joints(smpl_model, betas, shift_pelvis=True, use_tight_bbox=False):
    pose = np.zeros((1, 72), dtype="float32")
    vertices, _, canonical_joints = smpl_model(pose, betas)

    canonical_joints = canonical_joints.astype('float32')
    pelvis_pos = canonical_joints[0].copy()
    if shift_pelvis:
        vertices -= pelvis_pos
        canonical_joints -= pelvis_pos
    
    canonical_bbox = (
        vertices_to_bbox(vertices) if use_tight_bbox
        else skeleton_to_bbox(canonical_joints)
    )
    
    return canonical_joints, canonical_bbox, pelvis_pos


def load_motion_weights(subject_dir, grid_size, smpl_scale=1.0):
    mweights = np.load(subject_dir / f"mweights_nearest_{grid_size}-1.npy")
    contained = np.load(subject_dir / f"contained_{grid_size}-1.npy")
    closest_dist = np.load(subject_dir / f"closest_dist_{grid_size}-1.npy")
    # closest_idx = np.load(subject_dir / f"closest_idx_{grid_size}.npy")
    # with (subject_dir / "canonical_bbox.pkl").open("rb") as f:
    #     canonical_bbox = pickle.load(f)
    
    is_close = (closest_dist < (cfg.mweights_mesh_tol * smpl_scale**2)) | contained
    mweights *= is_close.astype('float32').reshape(grid_size, grid_size, grid_size)
    mweights = gaussian_filter(mweights, cfg.mweights_gaussian_blur, mode='constant', truncate=50)

    bg_weight = np.clip(1 - np.sum(mweights, axis=0, keepdims=True), 0, 1)
    mweights = np.concatenate([mweights, bg_weight], axis=0) + 1e-6
    mweights = mweights / np.sum(mweights, axis=0, keepdims=True).clip(min=1e-6)

    # np.save("mweights.npy", mweights)
    # return mweights, canonical_bbox
    return mweights


def load_train_cameras(subject_dir):
    cameras_path = subject_dir / "cameras.pkl"
    with cameras_path.open("rb") as f:
        cameras = pickle.load(f)
        
    for frame_name in cameras.keys():
        K1 = cameras[frame_name]['intrinsics'].copy()
        K1[:2] *= cfg.resize_img_scale
        cameras[frame_name]['intrinsics_scaled'] = K1

        scale_obs = cfg.get("resize_img_scale_obs", cfg.resize_img_scale)
        K2 = cameras[frame_name]['intrinsics'].copy()
        K2[:2] *= scale_obs
        cameras[frame_name]['intrinsics_scaled_obs'] = K2

    return cameras


def load_train_mesh_infos(subject_dir, estimated=False):
    mesh_infos_path = (
        subject_dir / "mesh_infos.pkl" if not estimated
        else subject_dir / "estimated_mesh_infos.pkl"
    )
    with mesh_infos_path.open("rb") as f:
        mesh_infos = pickle.load(f)
    return mesh_infos


def get_mesh(smpl_model, dst_skel):
    smpl_scale = dst_skel["smpl_scale"]

    cnl_vertices, cnl_faces, cnl_joints = smpl_model(np.zeros_like(dst_skel["poses"]), dst_skel["avg_betas"])
    cnl_joints *= smpl_scale
    cnl_vertices *= smpl_scale
    pelvis_pos = np.copy(cnl_joints[0])
    cnl_joints -= pelvis_pos
    cnl_vertices -= pelvis_pos
    cnl_mesh = Trimesh(cnl_vertices, cnl_faces)

    vertices, faces, joints = smpl_model(dst_skel["poses"], dst_skel["betas"])
    vertices *= smpl_scale
    joints -= pelvis_pos
    vertices -= pelvis_pos
    mesh = Trimesh(vertices, faces)
    return mesh, cnl_mesh


def swap_estimated(x):
    estimated_found = False
    for k, v in x.items():
        if k.startswith("estimated_"):
            idx = k.find('_')
            paired_k = k[idx+1:]
            x[paired_k] = v
            estimated_found = True
    if not estimated_found:
        raise RuntimeError("swap_estimated called but estimated not found")


def subsample_observed_views(batch, N):
    keys = ['src_imgs', 'in_dst_Rs', 'in_dst_Ts', 'in_dst_posevecs_69', 'in_K', 'in_E']
    for key in keys:
        batch[key] = batch[key][:N]


# @staticmethod
    # def get_canonical_vol_coords(volume_size, cnl_bbox_min_xyz, cnl_bbox_max_xyz):
    #     vsize = torch.max((cnl_bbox_max_xyz - cnl_bbox_min_xyz)).item() / (volume_size - 1) + 1e-6

    #     # voxel_size = cfg.voxel_size
    #     min_x, min_y, min_z = cnl_bbox_min_xyz
    #     max_x, max_y, max_z = cnl_bbox_max_xyz
    #     zs = torch.arange(
    #         min_z, max_z+vsize, vsize, device=cnl_bbox_min_xyz.device
    #     )
    #     ys = torch.arange(
    #         min_y, max_y+vsize, vsize, device=cnl_bbox_min_xyz.device
    #     )
    #     xs = torch.arange(
    #         min_x, max_x+vsize, vsize, device=cnl_bbox_min_xyz.device
    #     )
    #     Z, Y, X = torch.meshgrid(zs, ys, xs, indexing="ij")

    #     grid = torch.stack((X, Y, Z), dim=-1)
    #     return grid
    
def get_canonical_vol_coords(volume_size, canonical_bbox):
    min_x, min_y, min_z = canonical_bbox['min_xyz']
    max_x, max_y, max_z = canonical_bbox['max_xyz']
    zs = np.linspace(min_z, max_z, volume_size//2)
    ys = np.linspace(min_y, max_y, volume_size)
    xs = np.linspace(min_x, max_x, volume_size)
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")

    # bbox_center_xyz = (cnl_bbox_max_xyz + cnl_bbox_min_xyz) / 2
    # bbox_size = torch.max(cnl_bbox_max_xyz - cnl_bbox_min_xyz) / 2
    # coords = torch.linspace(-1, 1, self.volume_size).to(bbox_center_xyz)
    # X, Y, Z = torch.meshgrid(coords, coords, coords, indexing='ij')
    grid = np.stack((X, Y, Z), axis=-1)
    # can_vol_coords = bbox_center_xyz + grid * bbox_size
    return grid
