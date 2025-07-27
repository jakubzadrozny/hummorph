import math
import sys
from argparse import ArgumentParser
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from tools.prepare_dna.prepare_dataset import prepare_dir, prepare_sequence


def parse_args():
    parser = ArgumentParser(description='Prepare DNA-Rendering dataset')
    parser.add_argument('data_root', type=str, help='Root directory of the dataset')
    parser.add_argument('part', type=int)
    parser.add_argument('out_root', type=str, help='Output root directory')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    return parser.parse_args()


def main():
    logger.remove()
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
    logger.add(sys.stderr, level="INFO", format=log_format, colorize=True, backtrace=True, diagnose=True)
    logger.add("prepare_dna_rendering.log", level="INFO", format=log_format, colorize=False, backtrace=True, diagnose=True)

    args = parse_args()
    data_root = Path(args.data_root) / f"Part_{args.part}"
    out_root = prepare_dir(Path(args.out_root))
    
    rank = args.rank
    world_size = args.world_size
    # torch.set_num_threads(8)
    
    data_root_main = data_root / f"dna_rendering_part{args.part}_main"
    data_root_annots = data_root / f"dna_rendering_part{args.part}_annotations"
    all_sequences = sorted(
        [path.stem for path in data_root_main.glob("*.smc")]
    )

    chunk = math.ceil(len(all_sequences) / world_size)
    from_idx = rank*chunk
    to_idx = (rank+1)*chunk
    sequences = all_sequences[from_idx:to_idx]
    logger.info(f"Sequences to prepare: from {from_idx+1} to {to_idx+1} out of {len(all_sequences)} ({sequences})")
    
    for sequence in tqdm(sequences):
        prepare_sequence(
            sequence=sequence,
            data_root_main=data_root_main,
            data_root_annots=data_root_annots,
            out_root=out_root,
        )


if __name__ == '__main__':
    main()
