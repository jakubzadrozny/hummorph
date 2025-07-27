import torch.nn as nn  

class MotionWeightVolumeDecoder(nn.Module):
    def __init__(self, cfg, **_):
        super(MotionWeightVolumeDecoder, self).__init__()

    def forward(self, log_motion_weights_priors, **_):
        return log_motion_weights_priors
