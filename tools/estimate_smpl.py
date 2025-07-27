# based on: https://github.com/jeffffffli/HybrIK/blob/main/scripts/demo_image.py

import json
import os
import pickle
from pathlib import Path
from tqdm import tqdm

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict
from torch import multiprocessing as mp
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

from configs import cfg
from core.data.dataset_args import DatasetArgs
from core.data.hummorph.common import load_image_with_mask, load_train_cameras
from core.data.hummorph.is_train_subject_dir import is_train_subject_dir
from core.utils.image_util import to_8b_image
from third_parties.smpl.smpl_numpy import SMPL


MODEL_DIR = 'third_parties/smpl/models'

HYBRIK_CFG_PATH = 'configs/256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml'
HYBRIK_CKPT = 'pretrained_models/hybrik_hrnet48_w3dpw.pth'


def load_hybrik(hybrik_path):
    cwd = Path.cwd()
    os.chdir(str(hybrik_path))

    from hybrik.models import builder
    from hybrik.utils.config import update_config
    from hybrik.utils.presets import SimpleTransform3DSMPLCam

    hybrik_cfg = update_config(HYBRIK_CFG_PATH)
    
    bbox_3d_shape = getattr(hybrik_cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))
    bbox_3d_shape = [item * 1e-3 for item in bbox_3d_shape]
    dummpy_set = edict({
        'joint_pairs_17': None,
        'joint_pairs_24': None,
        'joint_pairs_29': None,
        'bbox_3d_shape': bbox_3d_shape
    })

    det_transform = T.Compose([T.ToTensor()])

    transformation = SimpleTransform3DSMPLCam(
        dummpy_set, scale_factor=hybrik_cfg.DATASET.SCALE_FACTOR,
        color_factor=hybrik_cfg.DATASET.COLOR_FACTOR,
        occlusion=hybrik_cfg.DATASET.OCCLUSION,
        input_size=hybrik_cfg.MODEL.IMAGE_SIZE,
        output_size=hybrik_cfg.MODEL.HEATMAP_SIZE,
        depth_dim=hybrik_cfg.MODEL.EXTRA.DEPTH_DIM,
        bbox_3d_shape=bbox_3d_shape,
        rot=hybrik_cfg.DATASET.ROT_FACTOR, sigma=hybrik_cfg.MODEL.EXTRA.SIGMA,
        train=False, add_dpg=False,
        loss_type=hybrik_cfg.LOSS['TYPE'],
    )

    det_model = fasterrcnn_resnet50_fpn(pretrained=True)

    hybrik_model = builder.build_sppe(hybrik_cfg.MODEL)

    print(f'Loading hybrik from {HYBRIK_CKPT}...')
    save_dict = torch.load(HYBRIK_CKPT, map_location='cpu')
    if type(save_dict) == dict:
        model_dict = save_dict['model']
        hybrik_model.load_state_dict(model_dict)
    else:
        hybrik_model.load_state_dict(save_dict)

    det_model.cuda()
    hybrik_model.cuda()
    det_model.eval()
    hybrik_model.eval()

    os.chdir(str(cwd))

    return det_model, det_transform, hybrik_model, transformation


class WorkerProcess(mp.Process):
    def __init__(
        self,
        rank: int,
        input_q: mp.Queue,
        dataset_path: Path,
        hybrik_root: Path,
        extension: str,
    ):
        super().__init__()
        self.rank = rank
        self.input_q = input_q
        self.dataset_path = dataset_path
        self.hybrik_root = hybrik_root
        self.extension = extension

    def run(self):
        from hybrik.utils.vis import get_one_box

        try:
            devices = list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
            cvd = str(devices[self.rank])
        except KeyError:
            cvd = str(self.rank)
        os.environ["CUDA_VISIBLE_DEVICES"] = cvd
        torch.set_num_threads(1)
        cv2.setNumThreads(1)

        smpl_model = SMPL(sex='neutral', model_dir=MODEL_DIR)

        det_model, det_transform, hybrik_model, transformation = load_hybrik(self.hybrik_root)

        bgcolor = np.array([255.0, 255.0, 255.0], dtype='float32')

        while True:
            subject_dir_name = self.input_q.get()
            # Exit the process
            if subject_dir_name is None:
                break

            subject_dir = self.dataset_path / subject_dir_name
            mesh_infos = {}

            frames = sorted((subject_dir / "images").glob(f"*.{self.extension}"))
            if len(frames) == 0:
                raise RuntimeError(f"No images found in {subject_dir}, wrong extension?")

            for frame in frames:
                frame_name = frame.stem
                cameras = load_train_cameras(subject_dir)
                img, _ = load_image_with_mask(
                    subject_dir,
                    frame_name,
                    cameras[frame_name],
                    bgcolor=bgcolor,
                    extension=self.extension,
                )
                input_image = to_8b_image(img)

                det_input = det_transform(input_image).to('cuda')
                with torch.inference_mode():
                    det_output = det_model([det_input])[0]
                tight_bbox = get_one_box(det_output)

                pose_input, bbox, img_center = transformation.test_transform(
                    input_image, tight_bbox
                )
                pose_input = pose_input.to('cuda')[None, :, :, :]
                # hybrik_model.focal_length = camera["intrinsics_scaled"][0, 0]
                with torch.inference_mode():
                    pose_output = hybrik_model(
                        pose_input, flip_test=True,
                        bboxes=torch.from_numpy(np.array(bbox)).to(pose_input.device).unsqueeze(0).float(),
                        img_center=torch.from_numpy(img_center).to(pose_input.device).unsqueeze(0).float()
                    )

                theta = pose_output['pred_theta_mats'].cpu().numpy().reshape(24, 3, 3)
                poses = np.zeros((24, 3))
                for i in range(1, 24):
                    poses[i, :] = cv2.Rodrigues(theta[i, :, :])[0][:, 0]

                poses = poses.reshape(1, 72)
                betas = pose_output['pred_shape'].cpu().numpy()[0]

                _, _, tpose_joints = smpl_model(np.zeros_like(poses), betas)
                _, _, joints = smpl_model(poses, betas)
                pelvis_pos = tpose_joints[0].copy()
                joints -= pelvis_pos
                tpose_joints -= pelvis_pos

                mesh_infos[frame_name] = {
                    'poses': poses,
                    'betas': betas,
                    'joints': joints, 
                    'tpose_joints': tpose_joints,
                }

            out_mesh_info_path = subject_dir / "estimated_mesh_infos.pkl"
            with out_mesh_info_path.open("wb") as f:
                pickle.dump(mesh_infos, f)


def main():
    dataset_name = cfg.train.dataset
    dataset_args = DatasetArgs.get(dataset_name)
    dataset_path = Path(dataset_args['dataset_path'])

    num_workers = cfg.n_gpus
    input_q: mp.Queue = mp.Queue(maxsize=2*num_workers)
    processes = [
        WorkerProcess(
            rank=idx,
            input_q=input_q,
            dataset_path=dataset_path,
            hybrik_root=Path(cfg.hybrik.root),
            extension=dataset_args['extension'],
        )
        for idx in range(num_workers)
    ]
    for proc in processes:
        proc.start()

    subject_dirs = [
        dir.stem for dir in dataset_path.iterdir() 
        if dir.is_dir() and (dir / "images").is_dir()
    ]

    if not cfg.hybrik.all:
        with (dataset_path / "splits.json").open() as splits_f:
            splits = json.load(splits_f)
        subjects = sorted(sum(list(splits.values()), []))

        # make sure your is_train_subject_dir makes sense in this case
        subject_dirs = [
            sub_dir for sub_dir in subject_dirs
            if is_train_subject_dir(sub_dir, subjects, "all")
        ]

    for subject_dir in tqdm(subject_dirs):
        # if not (dataset_path / subject_dir / "estimated_mesh_infos.pkl").exists():
        input_q.put(subject_dir)

    for _ in range(num_workers):
        input_q.put(None)

    for proc in processes:
        proc.join()


if __name__ == "__main__":
    main()
