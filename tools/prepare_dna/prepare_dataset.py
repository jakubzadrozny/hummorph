import pickle
import warnings
from pathlib import Path
from tqdm import tqdm

import cv2
import numpy as np
import smplx
from loguru import logger

from third_parties.smpl.smpl_numpy import SMPL
from third_parties.smpl.transfer_model.smplx_to_smpl import smplx_to_smpl, body_model_config
from third_parties.smpl.transfer_model.utils import read_deformation_transfer
from tools.prepare_dna.SMCReader import SMCReader


MODEL_DIR = Path("third_parties/smpl/models")


def prepare_dir(out_dir):
    out_dir.mkdir(exist_ok=True)
    return out_dir


def prepare_smpl(
    smplx_dict,
    num_frames,
    smpl_model, 
    smplx_model, 
    body_model, 
    def_matrix, 
    device=None,
):
    split_size = [1, smplx_model.NUM_BODY_JOINTS, 1, 1, 1, 15]
    split_ind = np.cumsum(split_size)
    chunks = np.split(
        smplx_dict["fullpose"],
        indices_or_sections=split_ind,
        axis=1,
    )
    global_orient = chunks[0]
    body_pose = chunks[1]
    smpl_scale = smplx_dict["scale"]
    smplx_betas = smplx_dict["betas"]
    if smplx_betas.shape[0] == 1:
        warnings.warn(
            "betas supplied only for one frame",
            category=RuntimeWarning,
            stacklevel=2,
        )
        T = smplx_dict["fullpose"].shape[0]
        smplx_betas = np.repeat(smplx_betas, T, axis=0)

    all_betas = []
    output = dict(
        mesh_infos={},
        smpl_params={},
        smplx_params={},
    )
    for idx in tqdm(range(num_frames)):
        out_name = 'frame_{:06d}'.format(idx)
        
        # load smplx params and convert them to smpl
        smplx_params = dict(
            betas=smplx_betas[idx:idx+1],
            transl=smplx_dict["transl"][idx:idx+1],
            body_pose=body_pose[idx:idx+1],
            global_orient=global_orient[idx:idx+1],
        )
        # print(f"Subject {subject}, frame {out_name}, begin transfer.")
        smpl_params = smplx_to_smpl(
            smplx_model=smplx_model, 
            body_model=body_model,
            def_matrix=def_matrix,
            smplx_params=smplx_params,
            device=device,
        )
        # print(f"Subject {subject}, frame {out_name}, transfer done.")
        output["smpl_params"][out_name] = smpl_params
        output["smplx_params"][out_name] = smplx_params
        
        betas = smpl_params['betas'][0] #(10,)
        poses = smpl_params['full_pose'][0].flatten()  #(72,)
        _, _, tpose_joints = smpl_model(np.zeros_like(poses), betas)
        # account for smpl_scale in DNA-Rendering dataset
        tpose_joints *= smpl_scale

        # get global Rh, Th
        Rh = poses[:3].copy()
        Th = tpose_joints[0].copy() # pelvis_pos
        
        all_betas.append(betas)

        # remove global rotation from body pose
        poses[:3] = 0
        _, _, joints = smpl_model(poses, betas)
        # account for smpl_scale in DNA-Rendering dataset
        joints *= smpl_scale

        # joint 0 should be at 0
        joints -= Th
        tpose_joints -= Th
        output["mesh_infos"][out_name] = {
            'Rh': Rh,
            'Th': Th + smpl_params["transl"][0] * smpl_scale,
            'poses': poses,
            'betas': betas,
            'joints': joints, 
            'tpose_joints': tpose_joints,
            'smpl_scale': smpl_scale,
        }

    avg_betas = np.mean(np.stack(all_betas, axis=0), axis=0)
    _, _, canonical_joints = smpl_model(np.zeros(72), avg_betas)
    # account for smpl_scale in DNA-Rendering dataset
    canonical_joints *= smpl_scale
    output["canonical_joints"] = canonical_joints
    return output


def prepare_view(rd_main, rd_annots, num_frames, out_dir, view, smpl_output):        
    out_img_dir  = prepare_dir(out_dir / 'images')
    out_mask_dir = prepare_dir(out_dir / 'masks')
    out_smpl_dir = prepare_dir(out_dir / 'smpl')
    out_smplx_dir = prepare_dir(out_dir / 'smplx')

    cam = rd_annots.get_Calibration(view)
    K = cam['K'].astype('float32') #(3, 3)
    D = cam['D'].astype('float32')
    c2w_E = cam['RT']

    # we need w2c
    E = np.linalg.inv(c2w_E)

    cameras = {}
    for idx in tqdm(range(num_frames)):
        out_name = 'frame_{:06d}'.format(idx)
        cameras[out_name] = dict(
            intrinsics=K,
            extrinsics=E,
            distortions=D,
        )

        out_smpl_path = out_smpl_dir / f"{out_name}.npz"
        smpl_params = smpl_output["smpl_params"][out_name]
        np.savez(str(out_smpl_path), **smpl_params)

        out_smplx_path = out_smplx_dir / f"{out_name}.npz"
        smplx_params = smpl_output["smplx_params"][out_name]
        np.savez(str(out_smplx_path), **smplx_params)

        img = rd_main.get_img('Camera_5mp', view, 'color', Frame_id=idx)
        out_image_path = out_img_dir / f"{out_name}.jpg"
        cv2.imwrite(str(out_image_path), img)
        # save_image(img, str(out_image_path))
    
        mask = rd_annots.get_mask(view, Frame_id=idx)
        out_mask_path = out_mask_dir / f"{out_name}.png"
        cv2.imwrite(str(out_mask_path), mask)
    
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


def load_smplx_model(gender):
    return smplx.create(
        model_path=str(MODEL_DIR),
        model_type="smplx",
        gender=gender,
        use_pca=False,
        num_betas=10,
    )


def find_processed_views(sequence, rd_main, out_root):
    info = rd_main.get_Camera_5mp_info()
    processed_view = None
    all = True
    for view in range(info["num_device"]):
        out_canonical_joints_path = out_root / f"{sequence}_{view}" / "canonical_joints.pkl"
        if out_canonical_joints_path.exists():
            processed_view = view
        else:
            all = False

    return all, processed_view


def load_smpl_from_view(out_root, sequence, view, num_frames):
    view_dir = out_root / f"{sequence}_{view}"

    output = dict(
        mesh_infos={},
        smpl_params={},
        smplx_params={},
    )
    for idx in range(num_frames):
        frame_name = 'frame_{:06d}'.format(idx)
        
        smpl_path = view_dir / 'smpl' / f"{frame_name}.npz"
        output["smpl_params"][frame_name] = dict(np.load(str(smpl_path), allow_pickle=True))
        
        smplx_path = view_dir / 'smplx' / f"{frame_name}.npz"
        output["smplx_params"][frame_name] = dict(np.load(str(smplx_path), allow_pickle=True))

    mesh_info_path = view_dir / "mesh_infos.pkl"
    with mesh_info_path.open("rb") as f:
        output["mesh_infos"] = pickle.load(f)

    canonical_joints_path = view_dir / "canonical_joints.pkl"
    with canonical_joints_path.open("rb") as f:
        output["canonical_joints"] = pickle.load(f)

    return output


def prepare_sequence(sequence, data_root_main, data_root_annots, out_root, device=None):
    logger.info(f"Begin processing sequence {sequence}")

    rd_main = SMCReader(data_root_main / f"{sequence}.smc")
    rd_annots = SMCReader(data_root_annots / f"{sequence}_annots.smc")
    num_frames = rd_main.get_Camera_5mp_info()["num_frame"]
    seq_ready, processed_view = find_processed_views(sequence, rd_main, out_root)
    if seq_ready:
        logger.info(f"Sequence {sequence} already processed")
        return
    elif processed_view is not None:
        logger.info(f"View {processed_view} for seq {sequence} already processed, loading...")
        smpl_output = load_smpl_from_view(out_root, sequence, processed_view, num_frames)
    else:
        smpl_model = SMPL(sex="neutral", model_dir=MODEL_DIR)
        gender = rd_main.actor_info["gender"]
        smplx_model = load_smplx_model(gender)

        cfg = body_model_config(smpl_models_path=MODEL_DIR)
        body_model = smplx.build_layer(str(MODEL_DIR), **cfg.body_model)
        if device is not None:
            body_model = body_model.to(device)

        deformation_transfer_path = MODEL_DIR / "model_transfer" / "smplx2smpl_deftrafo_setup.pkl"
        def_matrix = read_deformation_transfer(str(deformation_transfer_path), device=device)

        smplx_dict = rd_annots.get_SMPLx()
        smpl_output = prepare_smpl(
            smplx_dict=smplx_dict,
            num_frames=num_frames,
            smpl_model=smpl_model, 
            smplx_model=smplx_model, 
            body_model=body_model, 
            def_matrix=def_matrix,
            device=device,
        )

    info = rd_main.get_Camera_5mp_info()
    for view in range(info["num_device"]):
        out_dir = prepare_dir(out_root / f"{sequence}_{view}")
        prepare_view(
            rd_main=rd_main,
            rd_annots=rd_annots,
            num_frames=num_frames,
            view=str(view),
            out_dir=out_dir,
            smpl_output=smpl_output,
        )

    logger.info(f"Done processing sequence {sequence}")
