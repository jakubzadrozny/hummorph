import sys
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm
from loguru import logger

from tools.prepare_humman.prepare_dataset import prepare_dir, prepare_subject



def parse_args():
    parser = ArgumentParser(description='Prepare HuMMan dataset')
    parser.add_argument('dataset_root', type=str, help='root directory of the HuMMan dataset')
    parser.add_argument('out_root', type=str, help='output root directory')
    return parser.parse_args()


def main():
    logger.remove()
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
    logger.add(sys.stderr, level="INFO", format=log_format, colorize=True, backtrace=True, diagnose=True)
    logger.add("prepare_genebody.log", level="INFO", format=log_format, colorize=False, backtrace=True, diagnose=True)

    args = parse_args()
    data_root = Path(args.dataset_root)
    out_root = prepare_dir(Path(args.out_root))
    
    subject_dirs = sorted([
        subdir for subdir in data_root.iterdir() if subdir.is_dir()
    ])
    logger.info(f"Subjects to prepare: {len(subject_dirs)}")

    for subject_dir in tqdm(subject_dirs):
        prepare_subject(
            subject_dir=subject_dir,
            out_root=out_root,
        )


if __name__ == '__main__':
    main()
