import os
from pathlib import Path
from tqdm import tqdm

import cv2
import pandas as pd
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch import multiprocessing as mp
from skimage.metrics import structural_similarity

from configs import cfg
from core.data import create_dataset
from core.nets import create_network
from core.data.hummorph.common import swap_estimated
from core.utils.body_util import plot_pose
from core.utils.image_util import crop_img_to_bbox, to_8b_image, unpack_to_image
from core.utils.network_util import set_requires_grad
from core.utils.train_util import cpu_data_to_gpu
from core.train.trainers.hummorph.trainer import (
    Trainer,
    scale_for_lpips,
    EXCLUDE_KEYS_TO_GPU,
)
from third_parties.lpips import LPIPS


def psnr_metric(img_pred, img_gt):
    mse = np.mean((img_pred - img_gt)**2)
    psnr = -10 * np.log(mse) / np.log(10)
    return psnr


def ssim_metric(pred, target):
    return structural_similarity(pred, target, multichannel=True)


def lpips_metric(lpips, pred, target):
    pred_tensor = torch.from_numpy(pred).float().unsqueeze(0).cuda()
    target_tensor = torch.from_numpy(target).float().unsqueeze(0).cuda()
    with torch.no_grad():
        return (
            torch.mean(
                lpips(
                    scale_for_lpips(pred_tensor.permute(0, 3, 1, 2)),
                    scale_for_lpips(target_tensor.permute(0, 3, 1, 2)),
                )
            )
            .cpu()
            .item()
        )


def to_tensors(frame_data):
    for k, v in frame_data.items():
        if isinstance(v, np.ndarray):
            frame_data[k] = torch.from_numpy(v)
    return frame_data


class WorkerProcess(mp.Process):
    def __init__(
        self,
        rank: int,
        input_q: mp.Queue,
        output_q: mp.Queue,
        ckpt_path: Path,
        out_img_dir: Path,
    ):
        super().__init__()
        self.rank = rank
        self.input_q = input_q
        self.output_q = output_q
        self.ckpt_path = ckpt_path
        self.out_img_dir = out_img_dir

    def run(self):
        try:
            devices = list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
            cvd = str(devices[self.rank])
        except KeyError:
            cvd = str(self.rank)
        os.environ["CUDA_VISIBLE_DEVICES"] = cvd
        torch.set_num_threads(1)

        model = create_network()

        ckpt = torch.load(str(self.ckpt_path), map_location="cuda")
        model.load_state_dict(ckpt["network"], strict=True)
        model = model.cuda()
        model = model.eval()
        cfg.perturb = 0

        lpips = LPIPS(net="vgg").cuda()
        set_requires_grad(lpips, requires_grad=False)

        while True:
            batch = self.input_q.get()
            # Exit the process
            if batch is None:
                break

            # only access the first batch as we process one image one time
            for k, v in batch.items():
                batch[k] = v[0]

            frames_in_seq = batch["frames_in_seq"].item()
            frame_idx = batch["frame_idx"].item()
            eval_frames_from = np.clip(int(frames_in_seq * cfg.eval.cutoff + 0.5), 1, frames_in_seq) - cfg.eval.offset
            if frame_idx < eval_frames_from:
                self.output_q.put(None)
                continue
            
            if cfg.use_estimated_smpl:
                swap_estimated(batch)

            frame_data = cpu_data_to_gpu(batch, exclude_keys=EXCLUDE_KEYS_TO_GPU)

            with torch.inference_mode():
                net_output = model(batchify=True, **frame_data)

            width = batch["img_width"]
            height = batch["img_height"]
            ray_mask = batch["ray_mask"].cpu().numpy()

            rendered_img, _, truth_img = unpack_to_image(
                width=width,
                height=height,
                ray_mask=ray_mask,
                bgcolor=np.array(cfg.bgcolor) / 255.0,
                rgb=net_output["rgb"].cpu().numpy(),
                alpha=net_output["alpha"].cpu().numpy(),
                truth=batch["target_rgbs"],
                # crop_to_bbox=cfg.crop_to_bbox,
            )

            if cfg.crop_to_bbox:
                render_cropped, gt_cropped = crop_img_to_bbox(
                    [rendered_img, truth_img],
                    bbox=batch["bbox_2d"].numpy(),
                    resize_to=None,
                )
                pred_img_norm = render_cropped / 255.0
                gt_img_norm = gt_cropped / 255.0
            else:
                pred_img_norm = rendered_img / 255.0
                gt_img_norm = truth_img / 255.0

            metrics = dict(
                subject=batch["subject"],
                frame_name=batch["frame_name"],
                psnr=psnr_metric(pred_img_norm, gt_img_norm),
                ssim=ssim_metric(pred_img_norm, gt_img_norm),
                lpips=lpips_metric(lpips, pred_img_norm, gt_img_norm),
            )
            print(metrics)
            self.output_q.put(metrics)

            if cfg.write_imgs:
                img_dir = self.out_img_dir / batch["subject"]
                img_dir.mkdir(exist_ok=True)
                raw_img_dir = img_dir / "raw"
                raw_img_dir.mkdir(exist_ok=True)

                obs_imgs = to_8b_image(batch["src_imgs"].detach().cpu().numpy())
                num_obs = obs_imgs.shape[0]

                rendered_path = raw_img_dir / f"{batch['frame_name']}_render.png"
                cv2.imwrite(str(rendered_path), cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR))
                # gt_path = raw_img_dir / f"{batch['frame_name']}_gt.png"
                # cv2.imwrite(str(gt_path), cv2.cvtColor(truth_img, cv2.COLOR_RGB2BGR))
                # for i in range(num_obs):
                #     obs_path = raw_img_dir / f"obs_{i}.png"
                #     cv2.imwrite(str(obs_path), cv2.cvtColor(obs_imgs[i], cv2.COLOR_RGB2BGR))

                if cfg.plot_pose:
                    joints_2d = batch["joints_2d"].cpu().numpy().astype("int")
                    rendered_img = plot_pose(rendered_img, joints_2d)
                    truth_img = plot_pose(truth_img, joints_2d)

                    obs_joints_2d = batch["in_joints_2d"].cpu().numpy().astype("int")
                    for i in range(num_obs):
                        obs_imgs[i] = plot_pose(obs_imgs[i], obs_joints_2d[i])

                if cfg.crop_to_bbox:
                    # img_height, img_width = rendered_img.shape[:2]
                    # _ray_mask = ray_mask.reshape(img_height, img_width)
                    # rendered_img = 0.5 * rendered_img + 0.5 * _ray_mask[:, :, None] * np.array([0, 0, 255])[None, None, :]
                    rendered_img, truth_img = crop_img_to_bbox(
                       [rendered_img, truth_img],
                        bbox=batch["bbox_2d"].numpy(),
                    )
                    obs_img_height, obs_img_width = obs_imgs.shape[2:]
                    obs_bbox = [0, 0, obs_img_width, obs_img_height]
                    _obs_imgs = obs_imgs
                    obs_imgs = []
                    for i in range(num_obs):
                        obs_imgs.append(
                            crop_img_to_bbox([_obs_imgs[i]], bbox=obs_bbox)[0]
                        )

                height, width, _ = rendered_img.shape
                aspect_ratio = height / width

                # fig, axs = plt.subplots(2, 2, figsize=(14, 14 * aspect_ratio))
                num_imgs = 2+num_obs
                fig, axs = plt.subplots(1, num_imgs, figsize=(7 * num_imgs / aspect_ratio, 7))
                fig.set_tight_layout(True)

                for i in range(num_imgs):
                    axs[i].axis("off")

                axs[0].imshow(rendered_img)
                axs[0].set_title("rendered image")
                axs[1].imshow(truth_img)
                axs[1].set_title("ground truth")
                for i in range(num_obs):
                    axs[2+i].imshow(obs_imgs[i])
                    axs[2+i].set_title(f"observed image {i+1}")

                plt.savefig(img_dir / f"{batch['frame_name']}.png")
                plt.close(fig)


def main():
    # mp.set_sharing_strategy('file_system')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)

    cfg.chunk = 1024
    cfg.bgcolor = [0.0, 0.0, 0.0]

    ckpt_name = cfg.get("ckpt_name", cfg.experiment)
    ckpt_path = Path(Trainer.get_ckpt_path(ckpt_name))
    
    print(f"Using {cfg.n_gpus} GPUs")
    print(f"Using checkpoint {ckpt_name}")
    print(f"Using split {cfg.eval.split}")
    
    cfg_str = f"{ckpt_name}_resize-{cfg.resize_img_scale}_obs-{','.join([str(x) for x in cfg.eval.observed_frames])}"
    if cfg.eval.large:
        cfg_str += "_large"
    if cfg.crop_to_bbox:
        cfg_str += "_cropped"
    print(cfg_str)
    logdir = Path(cfg.logdir)
    out_img_dir = logdir / "eval" / cfg_str
    out_img_dir.mkdir(parents=True, exist_ok=True)

    num_workers = cfg.n_gpus
    input_q: mp.Queue = mp.Queue(maxsize=num_workers)
    output_q: mp.Queue = mp.Queue()
    processes = [
        WorkerProcess(
            rank=idx,
            input_q=input_q,
            output_q=output_q,
            ckpt_path=ckpt_path,
            out_img_dir=out_img_dir,
        )
        for idx in range(num_workers)
    ]
    for proc in processes:
        proc.start()

    results = []
    dataset = create_dataset(data_type="eval")
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        worker_init_fn=lambda _: cv2.setNumThreads(1),
        pin_memory=True,
    )

    for batch in tqdm(data_loader):
        input_q.put(batch)
    for _ in range(num_workers):
        input_q.put(None)

    for _ in range(len(dataset)):
        ret = output_q.get()
        if ret is not None:
            results.append(ret)

    for proc in processes:
        proc.join()

    results_df = pd.DataFrame(results)
    result_path = logdir / "eval" / f"{cfg.experiment}_{cfg_str}.csv"
    results_df.to_csv(result_path, index=False)
    print(f"written result to {str(result_path)}")

    print(f"{cfg_str} results:")
    print(results_df.groupby("subject").mean(numeric_only=True))
    return results_df


if __name__ == "__main__":
    main()
