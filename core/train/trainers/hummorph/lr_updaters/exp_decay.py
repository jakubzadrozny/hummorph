from configs import cfg

def get_customized_lr_names():
    return [k[3:] for k in cfg.train.keys() if k.startswith('lr_')]

def update_lr(optimizer, iter_step):
    decay_rate = 0.1
    decay_steps = cfg.train.lrate_decay * 1000
    decay_value = decay_rate ** (iter_step / decay_steps)
    for param_group in optimizer.param_groups:
        base_lr = (
            cfg.train[f"lr_{param_group['name']}"]
            if f"lr_{param_group['name']}" in cfg.train
            else cfg.train.lr
        )
        param_group['lr'] = base_lr * decay_value
