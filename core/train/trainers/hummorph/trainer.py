import os
import warnings
from shutil import copyfile

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.nn import functional as F

from configs import cfg
from core.train import create_lr_updater
from core.data import create_dataloader
from core.data.hummorph.common import swap_estimated
from core.utils.image_util import tile_images, to_8b_image
from core.utils.metric_accum import MetricAccum
from core.utils.network_util import set_requires_grad
from core.utils.train_util import cpu_data_to_gpu, Timer
from third_parties.lpips import LPIPS


img2mse = lambda x, y : torch.mean((x - y) ** 2)
img2l1 = lambda x, y : torch.mean(torch.abs(x-y))
to8b = lambda x : (255.*np.clip(x,0.,1.)).astype(np.uint8)


EXCLUDE_KEYS_TO_GPU = [
    'subject', 'subject_dir', 'frame_name', 'img_width',
    'img_height', 'sample_idx', 'obs_frame_ind', 'patch_info',
    'frame_idx', 'frames_in_seq',
]


def _unpack_imgs(rgbs, patch_masks, bgcolor, targets, div_indices):
    N_patch = len(div_indices) - 1
    assert patch_masks.shape[0] == N_patch
    assert targets.shape[0] == N_patch

    patch_imgs = bgcolor.expand(targets.shape).clone() # (N_patch, H, W, 3)
    for i in range(N_patch):
        patch_imgs[i, patch_masks[i]] = rgbs[div_indices[i]:div_indices[i+1]]

    return patch_imgs


def scale_for_lpips(image_tensor):
    return image_tensor * 2. - 1.


def get_mean_lr(optimizer):
    return np.mean(np.array(
        [param_group['lr'] for param_group in optimizer.param_groups]
    ))


def get_smoothness_kernel(sigma=1.0):
    xs = np.arange(-1, 2, 1)
    zgrid, ygrid, xgrid = np.meshgrid(xs, xs, xs, indexing='ij')
    grid = np.stack([xgrid, ygrid, zgrid], axis=-1)
    _sigma = np.array([sigma, sigma, 0.5*sigma])
    d0 = np.sum((1/_sigma)*grid**2, axis=-1)
    d1 = np.exp(-0.5 * d0)
    norm = np.sum(d1)
    G = d1 / norm
    G[1, 1, 1] -= 1
    return G


class Trainer(object):
    def __init__(self, network, optimizer, distributed_handler, device='cuda'):
        print('\n********** Init Trainer ***********')

        self.device = device
        self.distributed_handler = distributed_handler
        self.optimizer = optimizer
        self.network = network.to(device)
        self.update_lr = create_lr_updater()
        # self.scaler = torch.cuda.amp.GradScaler(enabled=False)

        self.clean = False

        self.local_accum = MetricAccum()
        self.timer = Timer()
        self.g_obs = torch.Generator().manual_seed(0)

        self.smoothness_kernel = torch.from_numpy(get_smoothness_kernel(0.5)).float().to(device)[None, None, ...]

        if "lpips" in cfg.train.lossweights.keys():
            self.lpips = LPIPS(net='vgg').to(device)
            set_requires_grad(self.lpips, requires_grad=False)

        print("Load Progress Dataset ...")
        self.prog_dataloader = create_dataloader(
            data_type='progress', 
            world_size=self.distributed_handler.ngpus, 
            rank=self.distributed_handler.rank,
        )[0]
        print('************************************')

        # load_ckpt has to go last to preserve RNG state loading
        if cfg.resume:
            if not Trainer.ckpt_exists(cfg.load_net):
                raise RuntimeError(f"checkpoint {cfg.load_net} does not exist")
            self.load_ckpt(f'{cfg.load_net}')
        else:
            self.iter = 0
            self.save_ckpt('init')
            self.iter = 1


    @staticmethod
    def get_ckpt_path(name):
        return os.path.join(cfg.logdir, f'{name}.pth')

    @staticmethod
    def get_log_code_dir():
        return os.path.join(cfg.logdir, 'code_log')

    @staticmethod
    def ckpt_exists(name):
        return os.path.exists(Trainer.get_ckpt_path(name))

    ######################################################3
    ## Training 

    def get_img_rebuild_loss(self, loss_names, rgb, target):
        losses = {}

        if "mse" in loss_names:
            losses["mse"] = img2mse(rgb, target)

        if "l1" in loss_names:
            losses["l1"] = img2l1(rgb, target)

        if "lpips" in loss_names:
            lpips_loss = self.lpips(scale_for_lpips(rgb.permute(0, 3, 1, 2)), 
                                    scale_for_lpips(target.permute(0, 3, 1, 2)))
            losses["lpips"] = torch.mean(lpips_loss)

        return losses

    def get_loss(self, net_output, pred_rgb_imgs, targets, mweights_dist_penalty):
        lossweights = cfg.train.lossweights
        loss_names = list(lossweights.keys())

        losses = self.get_img_rebuild_loss(loss_names, pred_rgb_imgs, targets)

        log_mweights_vol = net_output['log_mweights_vol']
        mweights_vol = torch.exp(log_mweights_vol)
        # entropy_loss = -torch.mean(torch.nansum(log_mweights_vol * mweights_vol, dim=0))
        dist_loss = torch.mean(mweights_vol.reshape(mweights_dist_penalty.shape) * mweights_dist_penalty)
        mweights_smoothness = F.conv3d(mweights_vol.unsqueeze(1), self.smoothness_kernel, padding='valid')
        smoothness_loss = torch.mean(mweights_smoothness**2)
        losses.update({
            # 'entropy': entropy_loss*1e6,
            'dist': dist_loss*1e3,
            'smoothness': smoothness_loss*1e3,
        })

        train_losses = {
            k: weight * losses[k] for k, weight in lossweights.items() if k in losses
        }

        if 'loss_consis' in net_output:
            loss_consis = net_output['loss_consis']
            if not torch.isnan(loss_consis):
                losses['consis'] = loss_consis
                train_losses['consis'] = loss_consis * lossweights['consis']

        return sum(train_losses.values()), losses
               

    def train_begin(self, train_dataloader):
        assert train_dataloader.batch_size == 1

        self.network.train()
        self.distributed_handler.broadcast(self.network, self.optimizer)
        cfg.perturb = cfg.train.perturb
        

    def train_end(self):
        pass

    def train(self, epoch, train_dataloader):
        self.train_begin(train_dataloader=train_dataloader)

        self.timer.begin()
        for batch_idx, batch in enumerate(train_dataloader):

            if self.iter > cfg.train.maxiter:
                break

            self.optimizer.zero_grad()

            # only access the first batch as we process one image one time
            for k, v in batch.items():
                batch[k] = v[0]

            using_estimated_params = False
            if cfg.use_estimated_smpl:
                estimated_swap_prob = np.minimum(
                    self.iter / (cfg.estimated_swap_warmup_kiter * 1000),
                    cfg.estimated_swap_prob_max,
                )
                if torch.rand(1).item() < estimated_swap_prob:
                    swap_estimated(batch)
                    using_estimated_params = True


            if batch['rays'].numel() == 0:
                warnings.warn("0 rays in batch")

            batch['iter_val'] = torch.full((1,), self.iter)
            data = cpu_data_to_gpu(
                batch, device=self.device, exclude_keys=EXCLUDE_KEYS_TO_GPU,
            )

            net_output = self.network(**data)
            if not net_output:
                warnings.warn("no net output")

            pred_rgb_imgs = _unpack_imgs(
                rgbs=net_output['rgb'],
                patch_masks=data['patch_masks'],
                bgcolor=data['bgcolor'] / 255.,
                targets=data['target_patches'],
                div_indices=data['patch_div_indices'],
            )
            train_loss, loss_dict = self.get_loss(
                net_output=net_output,
                pred_rgb_imgs=pred_rgb_imgs,
                targets=data['target_patches'],
                mweights_dist_penalty=data['mweights_dist_penalty'],
            )

            loss_dict["train_loss"] = train_loss
            train_loss.backward()

            grads = [
                param.grad.detach().flatten()
                for param in self.network.parameters()
                if param.grad is not None
            ]
            grads = torch.cat(grads) 
            loss_dict['grad_norm_2'] = grads.norm(2)
            loss_dict['grad_norm_inf'] = grads.norm(float('inf'))
            loss_dict['estimated_params'] = using_estimated_params
            self.local_accum.append(self.iter, loss_dict)
            
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 7.5)

            self.optimizer.step()

            if self.iter % cfg.train.log_interval == 0:
                mean_losses = self.local_accum.get_mean(cfg.train.log_interval)

                loss_str = f"Loss: {mean_losses['train_loss']:.4f} ["
                for k in ["mse", "lpips", "consis", "dist", "smoothness"]:
                    loss_str += f"{k}: {mean_losses[k]:.4f} "
                loss_str += "]"

                log_str = '(worker {worker}) Epoch: {epoch} [Iter {iter}, {batches_processed}/{batches_total} ({epoch_progress:.0f}%), {time}] {loss}, lr = {lr:.5f}, estimated = {estimated_params:.2f}'
                log_str = log_str.format(
                    worker=self.distributed_handler.rank,
                    epoch=epoch, 
                    iter=self.iter,
                    batches_processed=batch_idx, 
                    batches_total=len(train_dataloader),
                    epoch_progress=100. * batch_idx / len(train_dataloader), 
                    time=self.timer.log(),
                    loss=loss_str,
                    lr=get_mean_lr(self.optimizer),
                    estimated_params=mean_losses["estimated_params"],
                )
                print(log_str)

            is_reload_model = False
            if (self.iter in [1000, 3000, 5000]) or (self.iter % cfg.progress.dump_interval == 0):
                is_reload_model = self.progress()

            self.update_lr(self.optimizer, self.iter)
                    
            if not is_reload_model:
                if self.iter % cfg.train.save_checkpt_interval == 0:
                    self.save_ckpt('latest')
                if cfg.save_all:
                    if (self.iter in [1000, 3000, 5000] or 
                        self.iter % cfg.train.save_model_interval == 0):
                        self.save_ckpt(f'iter_{self.iter}')
                        # self.plot_losses()
        

            self.iter += 1
    
    def finalize(self):
        self.save_ckpt('latest')

    ######################################################3
    ## Progress

    def progress_begin(self):
        self.network.eval()
        cfg.perturb = 0.

    def progress_end(self):
        self.network.train()
        cfg.perturb = cfg.train.perturb

    def progress(self):
        self.progress_begin()

        print('Evaluate Progress Images ...')

        images = []
        is_empty_img = False
        for _, batch in enumerate(tqdm(self.prog_dataloader)):

            # only access the first batch as we process one image one time
            for k, v in batch.items():
                batch[k] = v[0]

            if cfg.use_estimated_smpl:
                swap_estimated(batch)

            width = batch['img_width']
            height = batch['img_height']
            ray_mask = batch['ray_mask']

            rendered = np.full(
                        (height * width, 3), np.array(cfg.bgcolor)/255., 
                        dtype='float32')
            truth = np.full(
                        (height * width, 3), np.array(cfg.bgcolor)/255., 
                        dtype='float32')

            batch['iter_val'] = torch.full((1,), self.iter)
            data = cpu_data_to_gpu(
                batch, 
                device=self.device, 
                exclude_keys=EXCLUDE_KEYS_TO_GPU + ['target_rgbs', 'in_vertex'],
            )
            with torch.no_grad():
                net_output = self.network(batchify=True, **data)

            rgb = net_output['rgb'].data.to("cpu").numpy()
            target_rgbs = batch['target_rgbs']

            rendered[ray_mask] = rgb
            truth[ray_mask] = target_rgbs

            truth = to_8b_image(truth.reshape((height, width, -1)))
            rendered = to_8b_image(rendered.reshape((height, width, -1)))
            images.append(np.concatenate([rendered, truth], axis=1))

             # check if we create empty images (only at the begining of training)
            if (self.iter <= 5000) and (np.allclose(rendered, np.array(cfg.bgcolor), atol=5.)):
                is_empty_img = True
                break

        tiled_image = tile_images(images)
        
        Image.fromarray(tiled_image).save(
            os.path.join(cfg.logdir, f"prog_{self.iter:06}_{self.distributed_handler.rank:02}.jpg")
        )

        if is_empty_img:
            print("Produce empty images; reload the init model.")
            # TODO: set self.iter correctly here
            # currently there is a problem with reproducibility in this case
            self.load_ckpt('init')
            
        self.progress_end()

        return is_empty_img


    ######################################################3
    ## Utils

    def save_ckpt(self, name):
        if self.distributed_handler.is_rank_0:
            path = Trainer.get_ckpt_path(name)
            print(f"Save checkpoint to {path} ...")

            torch.save({
                'iter': self.iter,
                'network': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, path)

        accum_state_name = f"{name}_accum_state_{self.distributed_handler.rank}.json"
        accum_path = os.path.join(cfg.logdir, accum_state_name)
        print(f"Save accum states to {accum_path} ...")
        with open(accum_path, "w+") as f:
            self.local_accum.dump(f)

        rng_ckpt_name = f"{name}_rng_states_{self.distributed_handler.rank}"
        rng_state_path = Trainer.get_ckpt_path(rng_ckpt_name)
        print(f"Save RNG states to {rng_state_path} ...")
        torch.save({
            'cpu': torch.random.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(),
            'obs': self.g_obs.get_state(),
        }, rng_state_path)


    def copy_ckpt(self, from_name, to_name):
        path_from = Trainer.get_ckpt_path(from_name)
        path_to = Trainer.get_ckpt_path(to_name)
        copyfile(path_from, path_to)

        for _rank in range(self.distributed_handler.ngpus):
            rng_from_path = Trainer.get_ckpt_path(f"{from_name}_rng_states_{_rank}")
            rng_to_path = Trainer.get_ckpt_path(f"{to_name}_rng_states_{_rank}")
            if os.path.exists(rng_from_path):
                copyfile(rng_from_path, rng_to_path)

            accum_path_from = os.path.join(cfg.logdir, f"{from_name}_accum_state_{_rank}.json")
            accum_path_to = os.path.join(cfg.logdir, f"{to_name}_accum_state_{_rank}.json")
            if os.path.exists(accum_path_from):
                copyfile(accum_path_from, accum_path_to)


    def load_ckpt(self, name):
        path = Trainer.get_ckpt_path(name)
        print(f"Load checkpoint from {path} ...")
        
        ckpt = torch.load(path, map_location=self.device)
        self.iter = ckpt['iter'] + 1
        self.losses_hist = []

        self.network.load_state_dict(ckpt['network'], strict=True)
        self.optimizer.load_state_dict(ckpt['optimizer'])

        accum_state_name = f"{name}_accum_state_{self.distributed_handler.rank}.json"
        accum_path = os.path.join(cfg.logdir, accum_state_name)
        if os.path.exists(accum_path):
            with open(accum_path) as f:
                self.local_accum.load(f)

        rng_ckpt_name = f"{name}_rng_states_{self.distributed_handler.rank}"
        if Trainer.ckpt_exists(rng_ckpt_name):
            rng_state_path = Trainer.get_ckpt_path(rng_ckpt_name)
            rng_ckpt = torch.load(rng_state_path)
            torch.set_rng_state(rng_ckpt['cpu'])
            torch.cuda.set_rng_state(rng_ckpt['cuda'])
            if 'obs' in rng_ckpt:
                self.g_obs.set_state(rng_ckpt['obs'])

        self.distributed_handler.broadcast(self.network, self.optimizer)
