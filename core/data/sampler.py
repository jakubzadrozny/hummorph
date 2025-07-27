from torch.utils.data.distributed import DistributedSampler


class DistributedSamplerWithSkip(DistributedSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip = 0

    def set_epoch(self, epoch: int, skip: int = 0) -> None:
        self.skip = skip
        super().set_epoch(epoch)

    def __iter__(self):
        iter_obj = super().__iter__()
        for _ in range(self.skip):
            next(iter_obj)
        return iter_obj

    def __len__(self):
        return super().__len__() - self.skip