import json
import pickle
from pathlib import Path
from shutil import copyfile
from tqdm import tqdm

import numpy as np
from loguru import logger

from third_parties.smpl.smpl_numpy import SMPL


MODEL_DIR = Path("third_parties/smpl/models")


def prepare_dir(out_dir):
    out_dir.mkdir(exist_ok=True)
    return out_dir


def prepare_smpl(param_paths, smpl_model):
    all_betas = []
    output = dict(
        mesh_infos={},
        smpl_params={},
    )
    for idx, param_path in enumerate(tqdm(param_paths)):
        out_name = 'frame_{:06d}'.format(idx)
        
        smpl_params = dict(np.load(str(param_path), allow_pickle=True))
        output["smpl_params"][out_name] = smpl_params
        
        betas = smpl_params['betas'] #(10,)
        poses = np.concatenate((smpl_params["global_orient"], smpl_params["body_pose"]))
        _, _, tpose_joints = smpl_model(np.zeros_like(poses), betas)
        
        # get global Rh, Th
        Rh = poses[:3].copy()
        Th = tpose_joints[0].copy() # pelvis_pos
        
        all_betas.append(betas)

        # remove global rotation from body pose
        poses[:3] = 0
        _, _, joints = smpl_model(poses, betas)
        
        # joint 0 should be at 0
        joints -= Th
        tpose_joints -= Th
        output["mesh_infos"][out_name] = {
            'Rh': Rh,
            'Th': Th + smpl_params["transl"],
            'poses': poses,
            'betas': betas,
            'joints': joints, 
            'tpose_joints': tpose_joints,
        }

    avg_betas = np.mean(np.stack(all_betas, axis=0), axis=0)
    _, _, canonical_joints = smpl_model(np.zeros(72), avg_betas)
    output["canonical_joints"] = canonical_joints
    return output


def prepare_view(sequence_dir, param_paths, out_dir, view, cam, smpl_output):
    out_img_dir  = prepare_dir(out_dir / 'images')
    out_mask_dir = prepare_dir(out_dir / 'masks')
    out_smpl_dir = prepare_dir(out_dir / 'smpl')

    K = np.array(cam['K'], dtype=np.float32)
    
    # we need w2c
    w2c_E = np.eye(4)  #(4, 4)
    w2c_R = np.array(cam['R'], dtype=np.float32)
    w2c_T = np.array(cam['T'], dtype=np.float32)
    w2c_E[:3, :3] = w2c_R
    w2c_E[:3, 3]= w2c_T

    E = w2c_E
    
    cameras = {}
    for idx, param_path in tqdm(enumerate(param_paths), total=len(param_paths)):
        out_name = 'frame_{:06d}'.format(idx)
        cameras[out_name] = dict(
            intrinsics=K,
            extrinsics=E,
        )

        out_smpl_path = out_smpl_dir / f"{out_name}.npz"
        smpl_params = smpl_output["smpl_params"][out_name]
        np.savez(str(out_smpl_path), **smpl_params)

        frame_name = param_path.stem
        img_path = sequence_dir / "kinect_color" / view / f"{frame_name}.png"
        out_image_path = out_img_dir / f"{out_name}.png"
        copyfile(str(img_path), str(out_image_path))

        mask_path = sequence_dir / "kinect_mask_manual" / view / f"{frame_name}.png"
        if not mask_path.exists():
            mask_path = sequence_dir / "kinect_mask" / view / f"{frame_name}.png"
        out_mask_path = out_mask_dir / f"{out_name}.png"
        copyfile(str(mask_path), str(out_mask_path))
    
    # write camera infos
    out_cameras_path = out_dir / "cameras.pkl"
    with out_cameras_path.open("wb") as f:
        pickle.dump(cameras, f)
    
    # write mesh infos
    out_mesh_info_path = out_dir / "mesh_infos.pkl"
    with out_mesh_info_path.open("wb") as f:
        pickle.dump(smpl_output["mesh_infos"], f)

    # write canonical joints
    out_canonical_joints_path = out_dir / "canonical_joints.pkl"
    with out_canonical_joints_path.open("wb") as f:
        pickle.dump({
            'joints': smpl_output["canonical_joints"],
        }, f)


def prepare_sequence(sequence_dir, out_root):
    smpl_params_dir = sequence_dir / "smpl_params"
    param_paths = list(sorted(smpl_params_dir.glob("*.npz")))
    
    smpl_model = SMPL(sex="neutral", model_dir=MODEL_DIR)
    
    logger.info(f"Begin processing subject {sequence_dir.stem}")

    smpl_output = prepare_smpl(
        param_paths=param_paths,
        smpl_model=smpl_model, 
    )

    cams_path = sequence_dir / 'cameras.json'
    with cams_path.open() as cams_f:
        cams = json.load(cams_f)
    
    for _view in cams.keys():
        color_pos = _view.rfind("_color_")
        if color_pos > -1:
            view = _view[:color_pos+1] + _view[color_pos+7:]
        else:
            continue

        out_dir = prepare_dir(out_root / f"{sequence_dir.stem}_{view}")
        prepare_view(
            sequence_dir=sequence_dir,
            param_paths=param_paths,
            out_dir=out_dir,
            view=view,
            cam=cams[_view],
            smpl_output=smpl_output,
        )

    logger.info(f"Done processing subject {sequence_dir.stem}")
