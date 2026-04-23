import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate deterministic SDD train/val split files.')
    parser.add_argument(
        '--data-root',
        default=r'C:\Users\giann\OneDrive\Desktop\SDD',
        help='Path to the SDD root containing kosXX folders.')
    parser.add_argument(
        '--out-dir',
        default='data/sdd/splits',
        help='Directory where split text files will be written.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-pos', type=int, default=41)
    parser.add_argument('--train-neg', type=int, default=82)
    parser.add_argument('--val-pos', type=int, default=11)
    parser.add_argument('--val-neg', type=int, default=70)
    return parser.parse_args()


def has_positive_pixels(mask_path):
    mask = np.array(Image.open(mask_path))
    return bool((mask > 0).any())


def write_split(path, samples):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(samples) + '\n', encoding='utf-8')


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)

    positives = []
    negatives = []
    for image_path in sorted(data_root.glob('kos*/Part*.jpg')):
        stem = image_path.with_suffix('')
        mask_path = image_path.with_name(stem.name + '_label.bmp')
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        rel = image_path.relative_to(data_root).with_suffix('').as_posix()
        if has_positive_pixels(mask_path):
            positives.append(rel)
        else:
            negatives.append(rel)

    rng = np.random.default_rng(args.seed)
    positives = rng.permutation(positives).tolist()
    negatives = rng.permutation(negatives).tolist()

    need_pos = args.train_pos + args.val_pos
    need_neg = args.train_neg + args.val_neg
    if len(positives) < need_pos:
        raise ValueError(f'Need {need_pos} positives, found {len(positives)}')
    if len(negatives) < need_neg:
        raise ValueError(f'Need {need_neg} negatives, found {len(negatives)}')

    train_pos = positives[:args.train_pos]
    train_neg = negatives[:args.train_neg]
    val_pos = positives[args.train_pos:args.train_pos + args.val_pos]
    val_neg = negatives[args.train_neg:args.train_neg + args.val_neg]
    train = train_pos + train_neg
    val = val_pos + val_neg

    rng.shuffle(train)
    rng.shuffle(val)

    write_split(out_dir / 'train_2x_pos.txt', train)
    write_split(out_dir / 'val_pos11_neg70.txt', val)
    write_split(out_dir / 'train_positive.txt', train_pos)
    write_split(out_dir / 'train_negative.txt', train_neg)
    write_split(out_dir / 'val_positive.txt', val_pos)
    write_split(out_dir / 'val_negative.txt', val_neg)

    print(f'Found {len(positives)} positive and {len(negatives)} negative samples')
    print(f'Wrote train_2x_pos.txt: {len(train_pos)} positive / {len(train_neg)} negative')
    print(f'Wrote val_pos11_neg70.txt: {len(val_pos)} positive / {len(val_neg)} negative')


if __name__ == '__main__':
    main()
