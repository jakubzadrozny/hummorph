from pathlib import Path
from shutil import copyfile

from configs import args, cfg
from core.utils.log_util import Logger
from core.data import create_dataloader
from core.nets import create_network
from core.train import create_trainer, create_optimizer
from core.train.trainers.hummorph.multigpu import HorovodMultiGPUHandler


def param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    distributed_handler = HorovodMultiGPUHandler(cfg.n_gpus)
    distributed_handler.set_seed(0) 
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    # torch.utils.deterministic.fill_uninitialized_memory = True
    log = Logger(distributed_handler=distributed_handler)
    if distributed_handler.is_rank_0:
        log.print_config()

    if not cfg.resume and distributed_handler.is_rank_0:
        logdir = Path(cfg.logdir)
        copyfile(args.cfg, str(logdir / "config.yaml"))

    train_loader, train_dataset, sampler = create_dataloader(
        data_type='train', 
        world_size=distributed_handler.ngpus, 
        rank=distributed_handler.rank,
    )

    num_subjects = len(train_dataset.subject_to_idx_map)
    model = create_network(num_subjects=num_subjects)

    if distributed_handler.is_rank_0:
        param_mil = param_count(model) / (10**6)
        print(f"Training model with {param_mil:.2f}M params")

    optimizer = create_optimizer(model, is_rank_0=distributed_handler.is_rank_0)
    optimizer = distributed_handler.distribute_optimizer(optimizer, model)
    
    trainer = create_trainer(model, optimizer, distributed_handler, device='cuda')

    # estimate start epoch
    epoch = (trainer.iter-1) // len(train_loader) + 1
    skip = trainer.iter - (epoch-1)*len(train_loader) - 1

    # re-seed dataloader workers
    for _ in range(epoch-1):
        iter(train_loader)

    while True:
        if trainer.iter > cfg.train.maxiter:
            break
        
        if distributed_handler.ngpus > 1:
            sampler.set_epoch(epoch, skip=skip)

        trainer.train(epoch=epoch, train_dataloader=train_loader)
        trainer.save_ckpt(f"epoch_{epoch}")

        epoch += 1
        skip = 0

    trainer.finalize()

if __name__ == '__main__':
    main()
