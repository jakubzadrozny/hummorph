from configs import cfg


def subject_to_name_and_view(subject):
    kinect_end = subject.rfind("_kinect_")
    if kinect_end > -1:
        name = subject[:kinect_end]
        view = int(subject[kinect_end+8:])
    else:
        name_end = subject.rfind("_")
        name = subject[:name_end]
        view = int(subject[name_end+1:])
    return name, view


def get_subject_front_view(subject):
    name, _ = subject_to_name_and_view(subject)
    if cfg.task == "humman":
        return f"{name}_kinect_000"
    elif cfg.task == "dna_rendering":
        return f"{name}_25"
    else:
        raise ValueError(f"unkown task {cfg.task}")


def is_train_subject_dir(subject_dir, subjects, split):
    name, view = subject_to_name_and_view(subject_dir)
    if split in ["train", "all"]:
        if cfg.task == "humman":
            view_ok = view not in [2, 7]
        elif cfg.task == "dna_rendering":
            view_ok = view % 4 == 1
        else:
            raise ValueError(f"unkown task {cfg.task}")
    elif split == "test":
        if cfg.task == "humman":
            view_ok = (view not in [2, 7]) if cfg.eval.large else (view == 0)
        elif cfg.task == "dna_rendering":
            view_ok = (view % 8 == 1) if cfg.eval.large else (view == 25)
        else:
            raise ValueError(f"unkown task {cfg.task}")
    else:
        raise ValueError(f"unkown split {split}")
    return (name in subjects) and view_ok
