import imp

from configs import cfg

import random
import numpy as np
import torch


def worker_init_fn(worker_id):
    random.seed(worker_id+100)
    np.random.seed(worker_id+100)
    torch.manual_seed(worker_id+100)


def _query_network():
    module = cfg.network_module
    module_path = module.replace(".", "/") + ".py"
    network = imp.load_source(module, module_path).Network
    return network


def create_network(num_subjects=1):
    network = _query_network()
    network = network(num_subjects=num_subjects)
    return network
