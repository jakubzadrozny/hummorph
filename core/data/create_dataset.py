import imp
import random

import cv2
import numpy as np
import torch

from core.data.sampler import DistributedSamplerWithSkip
from configs import cfg
from .dataset_args import DatasetArgs


def _query_dataset(data_type):
    module = cfg[data_type].dataset_module
    module_path = module.replace(".", "/") + ".py"
    dataset = imp.load_source(module, module_path).Dataset
    return dataset


def create_dataset(data_type='train'):
    dataset_name = cfg[data_type].dataset

    args = DatasetArgs.get(dataset_name)

    # customize dataset arguments according to dataset type
    args['bgcolor'] = None if data_type == 'train' else cfg.bgcolor

    split = cfg[data_type].get("split")
    if split is not None:
        args["split"] = split

    if data_type == "train":
        args['num_observed_frames'] = cfg[data_type].num_observed_frames
    if data_type == 'progress':
        args['max_subjects'] = 5
        args['max_frames'] = 10
    if data_type in ["progress", "eval", "tpose"]:
        args["obs_frame_ind"] = cfg[data_type].observed_frames
        max_frames = cfg[data_type].get("max_frames", None)
        if max_frames is not None:
            args["max_frames"] = max_frames

    dataset = _query_dataset(data_type)
    dataset = dataset(**args)
    return dataset


def _worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    # torch.manual_seed(worker_seed) -> WORKERS ARE NOT USING TORCH
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    cv2.setNumThreads(1)


def create_dataloader(data_type='train', world_size=1, rank=0):
    cfg_node = cfg[data_type]

    batch_size = cfg_node.batch_size
    shuffle = cfg_node.shuffle
    drop_last = cfg_node.drop_last

    dataset = create_dataset(data_type=data_type)
    g = torch.Generator()
    g.manual_seed(rank)
    if world_size > 1:
        train_sampler = DistributedSamplerWithSkip(
            dataset=dataset,
            num_replicas=world_size,
            rank=rank,
        )
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            drop_last=drop_last,
            num_workers=cfg.num_workers,
            worker_init_fn=_worker_init_fn,
            sampler=train_sampler,
            generator=g,
            pin_memory=True,
        )
    else:
        train_sampler = None
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=cfg.num_workers,
            worker_init_fn=_worker_init_fn,
            generator=g,
            pin_memory=True,
        )

    return data_loader, dataset, train_sampler
