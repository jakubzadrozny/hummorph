from configs import cfg

class DatasetArgs(object):
    dataset_attrs = {
        "humman_train": {
            "dataset_path": "/path/to/processed_humman",
            "split": "train",
            "keyfilter": cfg.train_keyfilter,
            "ray_shoot_mode": cfg.train.ray_shoot_mode,
            "extension": "png",
            # "maxframes": 100,
        },
        "humman_test": {
            "dataset_path": "/path/to/processed_humman",
            "split": "test",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image',
            "extension": "png",
        },
        "dna_rendering_train": {
            "dataset_path": "/path/to/processed_dna_rendering",
            "split": "train",
            "keyfilter": cfg.train_keyfilter,
            "ray_shoot_mode": cfg.train.ray_shoot_mode,
            # "maxframes": 100,
        },
        "dna_rendering_test": {
            "dataset_path": "/path/to/processed_dna_rendering",
            "split": "test",
            "keyfilter": cfg.test_keyfilter,
            "ray_shoot_mode": 'image',
            "max_frames": 15,
        },
    }

    @staticmethod
    def get(name):
        attrs = DatasetArgs.dataset_attrs[name]
        return attrs.copy()
