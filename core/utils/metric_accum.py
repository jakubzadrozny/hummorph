import json
from collections import defaultdict

import numpy as np
import torch


class MetricAccum:
    def __init__(self):
        self.accum = defaultdict(list)

    def append(self, iter, metrics):
        self.accum["iter"].append(iter)
        for k, v in metrics.items():
            if torch.is_tensor(v):
                v = torch.mean(v.detach().cpu()).item()
            self.accum[k].append(v)

    def get_mean(self, iters):
        return {
            k: np.mean(np.array(v[-iters:])) 
            for k, v in self.accum.items()
            if k != "iter"
        }
    
    def dump(self, f):
        json.dump(dict(self.accum), f)

    def load(self, f):
        self.accum = defaultdict(list, json.load(f))
