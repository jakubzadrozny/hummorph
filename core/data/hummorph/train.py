import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from matplotlib import pyplot as plt

from configs import cfg
from core.data.hummorph.common import (
    get_canonical_bbox,
    get_estimated_canonical_joints,
    get_mesh,
    load_image_with_mask,
    load_canonical_joints,
    load_motion_weights,
    load_train_cameras,
    load_train_mesh_infos,
    skeleton_to_bbox,
    vertices_to_bbox,
    get_canonical_vol_coords,
)
from core.data.hummorph.is_train_subject_dir import (
    is_train_subject_dir,
    subject_to_name_and_view,
)
from core.data.hummorph.patch_sampling import BadMaskException, sample_patch_rays
from core.utils.body_util import (
    approx_gaussian_bone_volumes, 
    body_pose_to_body_RTs,
    get_mweights_dist_penalty,
    get_canonical_global_tfms,
)
from core.utils.camera_util import (
    apply_global_tfm_to_camera,
    get_rays_from_KRT,
    project_3d_bbox,
    project_3d_points_to_camera_plane,
    rays_intersect_3d_bbox,
    get_render,
)
from third_parties.smpl.smpl_numpy import SMPL


MODEL_DIR = 'third_parties/smpl/models'


def query_dst_skeleton(mesh_infos, frame_name, estimated=False):
    keys = ["poses", "joints", "betas", "tpose_joints"]
    if not estimated:
        keys += ["Rh", "Th"]
    
    dst_skel = {
        key: np.array(mesh_infos[frame_name][key]).astype("float32")
        for key in keys
    }
    dst_skel["bbox"] = skeleton_to_bbox(mesh_infos[frame_name]["joints"])
    dst_skel["smpl_scale"] = mesh_infos[frame_name].get("smpl_scale", 1.0)
    
    if not estimated:
        dst_skel["avg_betas"] = np.mean(np.stack([
            np.array(mesh_infos[frame]["betas"])
            for frame in mesh_infos.keys()
        ], axis=0), axis=0).astype("float32")

    # smpl_path = self.subject_dirs[subject] / "smpl" / f"{frame_name}.npz"
    # dst_skel["betas"] = smpl_params["betas"][0].astype("float32")
    return dst_skel


def extract_observed_views(
    subject_dir, 
    subject_frame_names, 
    obs_frame_ind, 
    cameras, 
    mesh_infos,
    estimated_mesh_infos=None,
    extension="jpg",
):
    observed_views = defaultdict(list)
    for obs_idx in obs_frame_ind:
        if isinstance(obs_idx, str):
            if obs_idx == "mid":
                obs_idx = len(subject_frame_names) // 2
            else:
                raise ValueError("unknown frame code", obs_idx)
        elif isinstance(obs_idx, float) and obs_idx < 1:
            obs_idx = np.clip(int(len(subject_frame_names) * obs_idx + 0.5), 1, len(subject_frame_names)) - 1
        else:
            if obs_idx >= len(subject_frame_names):
                obs_idx = obs_idx % len(subject_frame_names)
        obs_frame_name = subject_frame_names[obs_idx]
        obs_camera = cameras[obs_frame_name]
        obs_skel_info = query_dst_skeleton(mesh_infos, obs_frame_name)

        _E = obs_camera['extrinsics']
        E = apply_global_tfm_to_camera(
            E=_E,
            Rh=obs_skel_info['Rh'],
            Th=obs_skel_info['Th']
        )
        observed_views["K"].append(obs_camera['intrinsics_scaled_obs'].astype('float32'))
        observed_views["E"].append(E.astype('float32'))
        
        pose = obs_skel_info['poses']
        joints = obs_skel_info['joints']
        tpose_joints = obs_skel_info['tpose_joints']
        observed_views["pose"].append(pose)
        observed_views["joints"].append(joints)
        observed_views["tpose_joints"].append(tpose_joints)

        dst_Rs, dst_Ts = body_pose_to_body_RTs(pose, tpose_joints)
        observed_views["dst_Rs"].append(dst_Rs)
        observed_views["dst_Ts"].append(dst_Ts)

        if estimated_mesh_infos is not None:
            obs_estimated_skel_info = query_dst_skeleton(estimated_mesh_infos, obs_frame_name, estimated=True)
            estimated_pose = obs_estimated_skel_info['poses']
            estimated_joints = obs_estimated_skel_info['joints']
            estimated_tpose_joints = obs_estimated_skel_info['tpose_joints']
            estimated_betas = obs_estimated_skel_info["betas"]
            observed_views["estimated_pose"].append(estimated_pose)
            observed_views["estimated_joints"].append(estimated_joints)
            observed_views["estimated_tpose_joints"].append(estimated_tpose_joints)
            observed_views["estimated_betas"].append(estimated_betas)

            estimated_dst_Rs, estimated_dst_Ts = body_pose_to_body_RTs(estimated_pose, estimated_tpose_joints)
            observed_views["estimated_dst_Rs"].append(estimated_dst_Rs)
            observed_views["estimated_dst_Ts"].append(estimated_dst_Ts)

        obs_img, _ = load_image_with_mask(
            subject_dir, 
            obs_frame_name, 
            obs_camera, 
            extension=extension, 
            scale=cfg.get("resize_img_scale_obs", cfg.resize_img_scale),
        )
        observed_views["img"].append(obs_img)

    return observed_views


class Dataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            dataset_path,
            keyfilter=None,
            obs_frame_ind=None,
            num_observed_frames=2,
            bgcolor=None,
            ray_shoot_mode='image',
            split='train',
            max_subjects=None,
            max_frames=None,
            extension="jpg",
            **_):

        print('[Dataset Path]', dataset_path)

        self.dataset_path = Path(dataset_path)
        self.obs_frame_ind = obs_frame_ind
        self.keyfilter = keyfilter
        self._bgcolor = bgcolor
        self.ray_shoot_mode = ray_shoot_mode
        self.extension = extension
        self.num_observed_frames = num_observed_frames

        with (self.dataset_path / "splits.json").open() as splits_f:
            self.splits = json.load(splits_f)
        self.splits["all"] = sorted(sum(list(self.splits.values()), []))
        self.subject_to_idx_map = dict(zip(
            self.splits["all"], range(len(self.splits["all"]))
        ))
        self.subjects = self.splits[split]
        self.smpl_model = SMPL(sex='neutral', model_dir=MODEL_DIR)

        if (self.dataset_path / "subject_dirs.json").exists():
            with (self.dataset_path / "subject_dirs.json").open() as f:
                all_subject_dirs = json.load(f)
        else:
            all_subject_dirs = [
                dir.stem for dir in self.dataset_path.iterdir() 
                if dir.is_dir() and (dir / "images").is_dir()
            ]
            with (self.dataset_path / "subject_dirs.json").open("w+") as f:
                json.dump(all_subject_dirs, f)
        self.subject_dirs = [
            sub_dir for sub_dir in all_subject_dirs
            if is_train_subject_dir(sub_dir, self.subjects, split)
        ]

        if max_subjects is not None:
            max_subjects = min(max_subjects, len(self.subject_dirs))
            subject_dirs = sorted(self.subject_dirs)
            skip = int(np.ceil(len(subject_dirs) / max_subjects))
            # self.subject_dirs = dict(random.sample(self.subject_dirs.items(), max_subjects))
            self.subject_dirs = subject_dirs[::skip]

        if (self.dataset_path / "subject_frames.json").exists():
            with (self.dataset_path / "subject_frames.json").open() as f:
                all_subject_frame_names = json.load(f)
        else:
            all_subject_frame_names = {
                subject_dir: sorted([
                    frame_path.stem for frame_path in (self.dataset_path / subject_dir / "images").glob(f"*.{self.extension}")
                ])
                for subject_dir in all_subject_dirs
            }
            with (self.dataset_path / "subject_frames.json").open("w+") as f:
                json.dump(all_subject_frame_names, f)
        self.subject_frame_names = {
            subject: all_subject_frame_names[subject]
            for subject in self.subject_dirs
        }
        
        if max_frames is not None:
            for subject, frame_names in self.subject_frame_names.items():
                skip = int(np.ceil(len(frame_names) / max_frames))
                self.subject_frame_names[subject] = frame_names[::skip]
        
        self.all_frame_names = sorted([
            (subject, frame_name)
            for subject in self.subject_dirs
            for frame_name in self.subject_frame_names[subject]
        ])
            
        print(f" -- Total Sequences: {len(self.subject_dirs)}")
        print(f' -- Total Frames: {len(self.all_frame_names)}')

    
    @property
    def bgcolor(self):
        return (
            np.array(self._bgcolor, dtype='float32') if self._bgcolor is not None
            else (np.random.rand(3) * 255.).astype('float32')
        )


    def get_tight_bbox(self, dst_skel, shift_pelvis=True):
        vertices, _, joints = self.smpl_model(dst_skel["poses"], dst_skel["betas"])
        
        smpl_scale = dst_skel["smpl_scale"]
        vertices *= smpl_scale
        joints *= smpl_scale

        joints = joints.astype('float32')
        pelvis_pos = joints[0].copy()
        if shift_pelvis:
            vertices -= pelvis_pos
        return vertices_to_bbox(vertices)
    

    def plot(self, img, dst_skel_info, H, W, K, E):
        mesh, _ = get_mesh(self.smpl_model, dst_skel_info)
        mesh_render = get_render(mesh, H, W, K, E)
        
        joints_3d = dst_skel_info["joints"]
        tpose_joints_3d = dst_skel_info["tpose_joints"]
        joints_2d = project_3d_points_to_camera_plane(joints_3d, K, E)
        # tpose_joints_2d = project_3d_points_to_camera_plane(tpose_joints_3d, K, E)
        
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
        axs[0].imshow(img)
        # axs[1].imshow(masks)
        axs[1].imshow(mesh_render)
        for ax in axs:
            ax.scatter(joints_2d[:, 0], joints_2d[:, 1], marker='x', s=8, c='r')
            # ax.scatter(tpose_joints_2d[:, 0], tpose_joints_2d[:, 1], marker='x', s=8, c='b')
        fig.tight_layout()
        plt.show()
        # plt.savefig(f"data_vis/{idx}.png", dpi=200)


    def sample_patches(self, results, dst_skel_info, img, subject_mask, rays_o, rays_d, prefix=''):
        H, W = img.shape[0:2]
        ray_img = img.reshape(-1, 3) 

        dst_bbox = dst_skel_info['bbox']
        near, far, ray_mask = rays_intersect_3d_bbox(dst_bbox, rays_o, rays_d)
        rays_o = rays_o[ray_mask]
        rays_d = rays_d[ray_mask]
        ray_img = ray_img[ray_mask]
        near = near[:, None].astype('float32')
        far = far[:, None].astype('float32')
        
        bbox_mask = ray_mask.reshape(H, W)
        subject_mask = np.bitwise_and(subject_mask, bbox_mask)

        if self.ray_shoot_mode == 'image':
            pass
        elif self.ray_shoot_mode == 'patch':
            try:
                rays_o, rays_d, ray_img, near, far, \
                target_patches, patch_masks, patch_div_indices, patch_info = sample_patch_rays(
                    img=img, 
                    H=H, 
                    W=W,
                    subject_mask=subject_mask,
                    bbox_mask=bbox_mask,
                    ray_mask=ray_mask,
                    rays_o=rays_o, 
                    rays_d=rays_d, 
                    ray_img=ray_img, 
                    near=near, 
                    far=far,
                )
            except BadMaskException:
                # print(f"Bad mask at idx {idx}, sub {subject}, frame {frame_name}", file=sys.stderr)
                subject_mask = bbox_mask
                rays_o, rays_d, ray_img, near, far, \
                target_patches, patch_masks, patch_div_indices, patch_info = \
                    sample_patch_rays(
                        img=img, 
                        H=H, 
                        W=W,
                        subject_mask=subject_mask,
                        bbox_mask=bbox_mask,
                        ray_mask=ray_mask,
                        rays_o=rays_o, 
                        rays_d=rays_d, 
                        ray_img=ray_img, 
                        near=near, 
                        far=far,
                    )
                
            patch_size = cfg.patch.size
            patch_sizes = patch_div_indices[1:] - patch_div_indices[:-1]
            small_patch = np.min(patch_sizes)
        else:
            assert False, f"Ivalid Ray Shoot Mode: {self.ray_shoot_mode}"

        batch_rays = np.stack([rays_o, rays_d], axis=0)

        if 'rays' in self.keyfilter:
            results.update({
                (prefix + 'ray_mask'): ray_mask,
                (prefix + 'rays'): batch_rays,
                (prefix + 'near'): near,
                (prefix + 'far'): far,
            })

            if self.ray_shoot_mode == 'patch':
                results.update({
                    (prefix + 'patch_div_indices'): patch_div_indices,
                    (prefix + 'patch_masks'): patch_masks,
                    (prefix + 'target_patches'): target_patches,
                })

        if 'target_rgbs' in self.keyfilter:
            results[prefix + 'target_rgbs'] = ray_img


    def add_motion_bases(self, results, poses, tpose_joints, canonical_joints, observed_views, prefix=''):
        Rs, Ts = body_pose_to_body_RTs(poses, tpose_joints)
        cnl_gtfms = get_canonical_global_tfms(canonical_joints)
        results.update({
            (prefix + 'dst_Rs'): Rs,
            (prefix + 'dst_Ts'): Ts,
            (prefix + 'cnl_gtfms'): cnl_gtfms,
            (prefix + 'in_dst_Rs'): np.stack(observed_views[prefix + "dst_Rs"], axis=0),
            (prefix + 'in_dst_Ts'): np.stack(observed_views[prefix + "dst_Ts"], axis=0),
        })


    def add_motion_weights(self, results, subject_dir, canonical_joints, canonical_bbox, vol_size=32, smpl_scale=1.0, prefix=''):
        if cfg.use_smpl_mweights:
            motion_weights = load_motion_weights(
                subject_dir=subject_dir,
                grid_size=cfg.mweight_volume.volume_size,
                smpl_scale=smpl_scale,
            )
        else:
            motion_weights = approx_gaussian_bone_volumes(
                canonical_joints,
                canonical_bbox['min_xyz'],
                canonical_bbox['max_xyz'],
                grid_size=cfg.mweight_volume.volume_size,
            )

        min_xyz = canonical_bbox['min_xyz'].astype('float32')
        max_xyz = canonical_bbox['max_xyz'].astype('float32')

        grid = get_canonical_vol_coords(vol_size, canonical_bbox)
        mweights_dist_penalty = get_mweights_dist_penalty(canonical_joints, grid.reshape(-1, 3))
        mweights_dist_penalty = np.concatenate(
            (mweights_dist_penalty, np.zeros((1, mweights_dist_penalty.shape[1]))), axis=0
        )

        results.update({
            (prefix + 'cnl_bbox_min_xyz'): min_xyz,
            (prefix + 'cnl_bbox_max_xyz'): max_xyz,
            (prefix + 'cnl_bbox_scale_xyz'): 2.0 / (max_xyz - min_xyz),
            (prefix + 'motion_weights_priors'): motion_weights.astype('float32'),
            (prefix + 'cnl_grid'): grid.astype('float32'),
            (prefix + 'mweights_dist_penalty'): mweights_dist_penalty.astype('float32'),
        })
        assert np.all(results['cnl_bbox_scale_xyz'] >= 0)


    def __len__(self):
        return len(self.all_frame_names)


    def __getitem__(self, idx):
        subject, frame_name = self.all_frame_names[idx]
        subject_dir = self.dataset_path / subject
        subject_name, _ = subject_to_name_and_view(subject)
        frame_idx = self.subject_frame_names[subject].index(frame_name)
        results = {
            'subject': subject,
            'frame_name': frame_name,
            'subject_idx': self.subject_to_idx_map[subject_name],
            'sample_idx': idx,
            'frame_idx': frame_idx,
            'frames_in_seq': len(self.subject_frame_names[subject]),
        }

        cameras = load_train_cameras(subject_dir)
        mesh_infos = load_train_mesh_infos(subject_dir)
        dst_skel_info = query_dst_skeleton(mesh_infos, frame_name)
        estimated_mesh_infos = (
            load_train_mesh_infos(subject_dir, estimated=True)
            if 'estimated_smpl' in self.keyfilter
            else None
        )

        if self.obs_frame_ind is None:
            non_target_frames = [
                idx 
                for idx in range(len(self.subject_frame_names[subject])) 
                if self.subject_frame_names[subject][idx] != frame_name
            ]
            obs_frame_ind = np.random.choice(
                np.array(non_target_frames), 
                size=self.num_observed_frames,
                replace=False,
            )
        else:
            obs_frame_ind = self.obs_frame_ind
        
        observed_views = extract_observed_views(
            subject_dir=subject_dir,
            subject_frame_names=self.subject_frame_names[subject],
            obs_frame_ind=obs_frame_ind,
            cameras=cameras,
            mesh_infos=mesh_infos,
            estimated_mesh_infos=estimated_mesh_infos,
            extension=self.extension,
        )

        camera = cameras[frame_name]
        bgcolor = self.bgcolor
        img, alpha = load_image_with_mask(subject_dir, frame_name, camera, bgcolor=bgcolor, extension=self.extension)

        K = camera['intrinsics_scaled'][:3, :3].astype('float32')
        _E = camera['extrinsics']
        E = apply_global_tfm_to_camera(
            E=_E,
            Rh=dst_skel_info['Rh'],
            Th=dst_skel_info['Th'],
        ).astype('float32')
        R = E[:3, :3]
        T = E[:3, 3]
        
        H, W = img.shape[0:2]
        rays_o, rays_d = get_rays_from_KRT(H, W, K, R, T)
        rays_o = rays_o.reshape(-1, 3) # (H, W, 3) --> (N_rays, 3)
        rays_d = rays_d.reshape(-1, 3)

        subject_mask = alpha[:, :, 0] > 0
        self.sample_patches(results, dst_skel_info, img, subject_mask, rays_o, rays_d)

        dst_poses = dst_skel_info['poses']
        canonical_joints, canonical_bbox = load_canonical_joints(
            subject_dir,
            shift_pelvis=cfg.shift_canonical_joints,
        )
        if cfg.use_tight_human_bbox:
            canonical_bbox = get_canonical_bbox(self.smpl_model, dst_skel_info)

        dst_tpose_joints = dst_skel_info['tpose_joints']
        joints = dst_skel_info["joints"]

        # self.plot(img, dst_skel_info, H, W, K, E)

        if 'rays' in self.keyfilter:
            results.update({
                'img_width': W,
                'img_height': H,
                'bgcolor': bgcolor, 
                # 'obs_frame_ind': obs_frame_ind,
                # 'rand': random.uniform(0, 1),
                # 'rand_np': np.random.rand(),
                # 'patch_info': patch_info['xy_min'],
                'ori_img': img,
                'src_imgs': np.stack(observed_views["img"], axis=0),
                'joints': joints,
                'canonical_joints': canonical_joints,
            })

        if 'motion_bases' in self.keyfilter:
            self.add_motion_bases(results, dst_poses, dst_tpose_joints, canonical_joints, observed_views)

        #print('len---in---dataset', len(in_dst_Rs))
        # get the bounding box of canonical volume
        if 'motion_weights_priors' in self.keyfilter or 'cnl_bbox' in self.keyfilter:
            self.add_motion_weights(results, subject_dir, canonical_joints, canonical_bbox, smpl_scale=dst_skel_info["smpl_scale"])
            
        if 'dst_posevec_69' in self.keyfilter:
            # 1. ignore global orientation
            # 2. add a small value to avoid all zeros
            dst_posevec_69 = dst_poses[3:] + 1e-2
            results['dst_posevec'] = dst_posevec_69

        if 'joints_2d' in self.keyfilter:
            results["joints_2d"] = project_3d_points_to_camera_plane(joints, K, E)
            results["in_joints_2d"] = np.stack([
                project_3d_points_to_camera_plane(
                    observed_views["joints"][i],
                    observed_views["K"][i],
                    observed_views["E"][i],
                ) for i in range(len(obs_frame_ind))
            ])
            tight_bbox = self.get_tight_bbox(dst_skel_info)
            results["bbox_2d"] = project_3d_bbox(tight_bbox, rays_o, rays_d, H, W)

        obs_dst_posevecs_69 = [poses[3:] + 1e-2 for poses in observed_views["pose"]]
        results["in_dst_posevecs_69"] = np.stack(obs_dst_posevecs_69, axis=0)

        results.update({
            'in_K': np.stack(observed_views["K"], axis=0),
            'in_E': np.stack(observed_views["E"], axis=0),
            'E': E,
            'K': K
        })

        if 'estimated_smpl' in self.keyfilter:
            estimated_avg_betas = np.mean(np.stack([
                np.array(betas)
                for betas in observed_views["estimated_betas"]
            ], axis=0), axis=0).astype("float32")
            estimated_canonical_joints, estimated_canonical_bbox, estimated_pelvis_pos = get_estimated_canonical_joints(
                self.smpl_model,
                estimated_avg_betas,
                shift_pelvis=cfg.shift_canonical_joints,
                use_tight_bbox=cfg.use_tight_human_bbox,
            )
            _, _, estimated_joints = self.smpl_model(dst_poses, estimated_avg_betas)
            estimated_joints -= estimated_pelvis_pos

            if 'rays' in self.keyfilter:
                results.update({
                    'estimated_joints': estimated_joints,
                    'estimated_canonical_joints': estimated_canonical_joints,
                })

            if 'motion_bases' in self.keyfilter:
                self.add_motion_bases(results, dst_poses, estimated_canonical_joints, estimated_canonical_joints, observed_views, prefix="estimated_")

            if 'motion_weights_priors' in self.keyfilter or 'cnl_bbox' in self.keyfilter:
                self.add_motion_weights(results, subject_dir, estimated_canonical_joints, estimated_canonical_bbox, prefix='estimated_')

            if 'joints_2d' in self.keyfilter:
                results["estimated_joints_2d"] = project_3d_points_to_camera_plane(estimated_joints, K, E)
                results["estimated_in_joints_2d"] = np.stack([
                    project_3d_points_to_camera_plane(
                        observed_views["estimated_joints"][i],
                        observed_views["K"][i],
                        observed_views["E"][i],
                    ) for i in range(len(obs_frame_ind))
                ])

            estimated_obs_dst_posevecs_69 = [poses[3:] + 1e-2 for poses in observed_views["estimated_pose"]]
            results["estimated_in_dst_posevecs_69"] = np.stack(estimated_obs_dst_posevecs_69, axis=0)    

        # results["dst_skel"] = dst_skel_info
        # results["skel_tmp"] = query_dst_skeleton(estimated_mesh_infos, frame_name, estimated=True)
        # results["skel_tmp"]["avg_betas"] = estimated_avg_betas
        return results
