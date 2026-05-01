import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Plot training metrics from an mmseg JSON log.')
    parser.add_argument('log_json', help='Path to the .log.json file')
    parser.add_argument('--out', default=None, help='Output PNG path')
    return parser.parse_args()


def load_points(path):
    points = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if 'iter' in row:
                points.append(row)
    return points


def main():
    args = parse_args()
    path = Path(args.log_json)
    out_path = Path(args.out) if args.out else path.with_suffix('.png')
    points = load_points(path)
    if not points:
        raise RuntimeError('No iteration records found in log json.')

    iters = [p['iter'] for p in points]
    loss = [p.get('loss') for p in points]
    acc = [p.get('decode.acc_seg') for p in points]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(iters, loss, label='loss')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(iters, acc, label='decode.acc_seg')
    axes[1].set_ylabel('Seg Acc')
    axes[1].set_xlabel('Iteration')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f'Saved plot to {out_path}')


if __name__ == '__main__':
    main()
