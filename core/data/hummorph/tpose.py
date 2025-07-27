import json
from pathlib import Path

import numpy as np
import torch
import torch.utils.data

from configs import cfg
from core.data.hummorph.common import (
    load_canonical_joints,
    load_motion_weights,
    load_train_cameras,
    load_train_mesh_infos,
)
from core.data.hummorph.train import extract_observed_views
from core.data.hummorph.is_train_subject_dir import (
    is_train_subject_dir,
    subject_to_name_and_view,
)
from core.utils.body_util import (
    body_pose_to_body_RTs,
    get_canonical_global_tfms,
    approx_gaussian_bone_volumes,
)
from core.utils.camera_util import (
    get_rays_from_KRT,
    project_3d_points_to_camera_plane,
    rays_intersect_3d_bbox,
    tpose_camera,
)
# from third_parties.smpl.smpl_numpy import SMPL

MODEL_DIR = 'third_parties/smpl/models'


class Dataset(torch.utils.data.Dataset):

    CAM_PARAMS = {
        'radius': 6.0, 'focal': 1250.
    }

    def __init__(
            self, 
            dataset_path,
            keyfilter=None,
            obs_frame_ind=None,
            bgcolor=None,
            ray_shoot_mode='image',
            split='train',
            max_subjects=None,
            extension="jpg",
            **_):

        print('[Dataset Path]', dataset_path)

        self.dataset_path = Path(dataset_path)
        self.obs_frame_ind = obs_frame_ind
        self.keyfilter = keyfilter
        self._bgcolor = bgcolor
        self.ray_shoot_mode = ray_shoot_mode
        self.extension = extension
        # self.smpl_model = SMPL(sex='neutral', model_dir=MODEL_DIR)

        with (self.dataset_path / "splits.json").open() as splits_f:
            self.splits = json.load(splits_f)
        self.splits["all"] = sorted(sum(list(self.splits.values()), []))
        self.subject_to_idx_map = dict(zip(
            self.splits["all"], range(len(self.splits["all"]))
        ))
        self.subjects = self.splits[split]

        with (self.dataset_path / "subject_dirs.json").open() as f:
            all_subject_dirs = json.load(f)
        self.subject_dirs = [
            sub_dir for sub_dir in all_subject_dirs
            if is_train_subject_dir(sub_dir, self.subjects, split)
        ]

        if max_subjects is not None:
            max_subjects = min(max_subjects, len(self.subject_dirs))
            subject_dirs = sorted(self.subject_dirs)
            skip = int(np.ceil(len(subject_dirs) / max_subjects))
            self.subject_dirs = subject_dirs[::skip]

        with (self.dataset_path / "subject_frames.json").open() as f:
            all_subject_frame_names = json.load(f)
        self.subject_frame_names = {
            subject: all_subject_frame_names[subject]
            for subject in self.subject_dirs
        }
        
        self.total_frames = len(self.subject_dirs) * cfg.tpose_render_frames
        print(f" -- Total Sequences: {len(self.subject_dirs)}")
        print(f' -- Total Frames: {self.total_frames}')


    def __len__(self):
        return self.total_frames


    @property
    def bgcolor(self):
        return (
            np.array(self._bgcolor, dtype='float32') if self._bgcolor is not None
            else (np.random.rand(3) * 255.).astype('float32')
        )

    # @staticmethod
    # def rotate_bbox(bbox, rmtx):
    #     min_x, min_y, min_z = bbox['min_xyz']
    #     max_x, max_y, max_z = bbox['max_xyz']

    #     bbox_pts = np.array(
    #         [[min_x, min_y, min_z],
    #          [min_x, min_y, max_z],
    #          [min_x, max_y, min_z],
    #          [min_x, max_y, max_z],
    #          [max_x, min_y, min_z],
    #          [max_x, min_y, max_z],
    #          [max_x, max_y, min_z],
    #          [max_x, max_y, max_z],])

    #     rotated_bbox_pts = bbox_pts.dot(rmtx)
    #     rotated_bbox = {
    #         'min_xyz': np.min(rotated_bbox_pts, axis=0),
    #         'max_xyz': np.max(rotated_bbox_pts, axis=0)
    #     }

    #     return rotated_bbox
    

    def __getitem__(self, idx):
        subject_idx = idx // cfg.tpose_render_frames
        subject = self.subject_dirs[subject_idx]
        subject_dir = self.dataset_path / subject
        
        frame_idx = idx % cfg.tpose_render_frames
        frame_name = 'frame_{:06d}'.format(frame_idx)

        subject_name, _ = subject_to_name_and_view(subject)
        results = {
            'subject': subject,
            'frame_name': frame_name,
            'sample_idx': idx,
            'subject_idx': self.subject_to_idx_map[subject_name],
        }
    
        cameras = load_train_cameras(subject_dir)
        mesh_infos = load_train_mesh_infos(subject_dir)
        joints, _ = load_canonical_joints(
            subject_dir,
            shift_pelvis=cfg.get("shift_canonical_joints", True),
        )
        canonical_joints, canonical_bbox = load_canonical_joints(
            subject_dir,
            shift_pelvis=cfg.get("shift_canonical_joints", True),
        )
        
        observed_views = extract_observed_views(
            subject_dir=subject_dir,
            subject_frame_names=self.subject_frame_names[subject],
            obs_frame_ind=self.obs_frame_ind,
            cameras=cameras,
            mesh_infos=mesh_infos,
            extension=self.extension,
        )

        # load t-pose
        dst_poses = np.zeros(72, dtype='float32')

        # rotate body
        angle = 2 * np.pi / cfg.tpose_render_frames * frame_idx
        # add_rmtx = cv2.Rodrigues(np.array([0, -angle, 0], dtype='float32'))[0]
        # root_rmtx = cv2.Rodrigues(dst_poses[:3])[0]
        # new_root_rmtx = add_rmtx@root_rmtx
        # dst_poses[:3] = cv2.Rodrigues(new_root_rmtx)[0][:, 0]
        K, E = tpose_camera(
            img_size=cfg.tpose_render_size,
            angle=angle,
            **self.CAM_PARAMS,
        )
        R = E[:3, :3]
        T = E[:3, 3]

        # rotate boundinig box
        # dst_bbox = self.rotate_bbox(dst_bbox, add_rmtx)
        
        W = H = cfg.tpose_render_size
        rays_o, rays_d = get_rays_from_KRT(H, W, K, R, T)
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)

        # (selected N_samples, ), (selected N_samples, ), (N_samples, )
        near, far, ray_mask = rays_intersect_3d_bbox(canonical_bbox, rays_o, rays_d)
        rays_o = rays_o[ray_mask]
        rays_d = rays_d[ray_mask]
        
        near = near[:, None].astype('float32')
        far = far[:, None].astype('float32')

        batch_rays = np.stack([rays_o, rays_d], axis=0) 

        if 'rays' in self.keyfilter:
            results.update({
                'img_width': W,
                'img_height': H,
                'ray_mask': ray_mask,
                'rays': batch_rays,
                'near': near,
                'far': far,
                'bgcolor': self.bgcolor, 
                'src_imgs': np.stack(observed_views["img"], axis=0),
                'joints': joints,
                'canonical_joints': canonical_joints
            })

        if 'motion_bases' in self.keyfilter:
            dst_Rs, dst_Ts = body_pose_to_body_RTs(dst_poses, joints)
            cnl_gtfms = get_canonical_global_tfms(canonical_joints)
            results.update({
                'dst_Rs': dst_Rs,
                'dst_Ts': dst_Ts,
                'cnl_gtfms': cnl_gtfms,
                'in_dst_Rs': np.stack(observed_views["dst_Rs"], axis=0),
                'in_dst_Ts': np.stack(observed_views["dst_Ts"], axis=0),
            })

        if 'motion_weights_priors' in self.keyfilter:
            if cfg.get("use_smpl_mweights", False):
                motion_weights = load_motion_weights(
                    subject_dir=subject_dir,
                    grid_size=cfg.mweight_volume.volume_size,
                    smpl_scale=mesh_infos["frame_000000"].get("smpl_scale", 1.0),
                )
            else:
                motion_weights = approx_gaussian_bone_volumes(
                    canonical_joints,   
                    canonical_bbox['min_xyz'],
                    canonical_bbox['max_xyz'],
                    grid_size=cfg.mweight_volume.volume_size,
                )
            results["motion_weights_priors"] = motion_weights.astype('float32'),

        # get the bounding box of canonical volume
        if 'cnl_bbox' in self.keyfilter:
            min_xyz = canonical_bbox['min_xyz'].astype('float32')
            max_xyz = canonical_bbox['max_xyz'].astype('float32')
            results.update({
                'cnl_bbox_min_xyz': min_xyz,
                'cnl_bbox_max_xyz': max_xyz,
                'cnl_bbox_scale_xyz': 2.0 / (max_xyz - min_xyz)
            })
            assert np.all(results['cnl_bbox_scale_xyz'] >= 0)

        if 'joints_2d' in self.keyfilter:
            results["joints_2d"] = project_3d_points_to_camera_plane(joints, K, E)
            results["in_joints_2d"] = np.stack([
                project_3d_points_to_camera_plane(
                    observed_views["joints"][i],
                    observed_views["K"][i],
                    observed_views["E"][i],
                ) for i in range(len(self.obs_frame_ind))
            ])

        if 'dst_posevec_69' in self.keyfilter:
            # 1. ignore global orientation
            # 2. add a small value to avoid all zeros
            dst_posevec_69 = dst_poses[3:] + 1e-2
            results['dst_posevec'] = dst_posevec_69

        obs_dst_posevecs_69 = [poses[3:] + 1e-2 for poses in observed_views["pose"]]
        results["in_dst_posevecs_69"] = np.stack(obs_dst_posevecs_69, axis=0)

        results.update({
            'in_K': np.stack(observed_views["K"], axis=0),
            'in_E': np.stack(observed_views["E"], axis=0),
            'E': E,
            'K': K
        })
        return results
