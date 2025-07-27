import os
import random
import numpy as np
import torch

try:
    import horovod.torch as hvd
    from mpi4py import MPI
    HAS_HOROVOD = True
except ImportError:
    hvd = None
    HAS_HOROVOD = False


class HorovodMultiGPUHandler:
    def __init__(self, ngpus=1):
        self.rank = 0
        self.local_rank = 0
        self.ngpus = ngpus
        if self.is_multi_gpu:
            if not HAS_HOROVOD:
                raise RuntimeError("Install horovod to run training on multiple GPUS.")
            hvd.init()
            try:
                devices = list(map(int, os.environ["CUDA_VISIBLE_DEVICES"].split(",")))
                cvd = str(devices[hvd.local_rank()])
            except KeyError:
                cvd = str(hvd.local_rank())

            os.environ["CUDA_VISIBLE_DEVICES"] = cvd
            # torch.multiprocessing.set_sharing_strategy('file_system')
            # torch.cuda.set_device(hvd.local_rank())
            torch.set_num_threads(1)
            self.local_rank = hvd.local_rank()
            self.rank = hvd.rank()

    def barrier(self):
        if self.is_multi_gpu:
            MPI.COMM_WORLD.barrier()

    def distribute_optimizer(self, optimizer, model):
        if self.is_multi_gpu:
            optimizer = hvd.DistributedOptimizer(
                optimizer, named_parameters=model.named_parameters()
            )
            self.broadcast(model, optimizer)
        return optimizer
    
    def broadcast(self, model, optimizer):
        if self.is_multi_gpu:
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            # hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    def set_seed(self, seed):
        seed = seed + self.rank
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    @property
    def is_rank_0(self):
        return self.rank == 0

    @property
    def is_multi_gpu(self):
        return self.ngpus > 1

    def if_rank_0(self, fn, *args, **kwargs):
        if self.is_rank_0:
            fn(*args, **kwargs)
