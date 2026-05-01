"""
Colab-ready SegFormer training script for Kolektor SDD.

Why this version:
- Uses the modern Hugging Face Transformers SegFormer implementation.
- Runs cleanly on Colab GPU without the older local MMSeg dependency stack.
- Preserves the same SDD setup we used locally:
  * resize to 512x1408
  * train split: 41 positive / 82 negative
  * val split: 11 positive / 70 negative
  * binary masks (background=0, positive=1)
  * custom validation metrics:
      pos_dice, pos_iou, neg_fp_rate, neg_fp_pixel_rate, sdd_score

Suggested Colab install cell:
!pip install -q transformers accelerate torch torchvision pillow matplotlib pandas tqdm

Example Colab run:
!python segformer_sdd_colab.py \
    --data-root /content/SDD \
    --output-dir /content/segformer_sdd_runs \
    --epochs 20 \
    --batch-size 4 \
    --model-name nvidia/segformer-b1-finetuned-ade-512-512
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


ID2LABEL = {0: "background", 1: "positive"}
LABEL2ID = {"background": 0, "positive": 1}


@dataclass
class TrainConfig:
    data_root: str
    output_dir: str
    model_name: str = "nvidia/segformer-b1-finetuned-ade-512-512"
    seed: int = 42
    image_height: int = 512
    image_width: int = 1408
    train_pos: int = 41
    train_neg: int = 82
    val_pos: int = 11
    val_neg: int = 70
    batch_size: int = 4
    eval_batch_size: int = 2
    epochs: int = 20
    lr: float = 6e-5
    weight_decay: float = 1e-2
    preview_samples: int = 2
    lambda_fp: float = 0.10


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_mask_binary(mask_path: Path) -> np.ndarray:
    mask = np.array(Image.open(mask_path))
    return (mask > 0).astype(np.uint8)


def discover_samples(data_root: Path) -> Tuple[List[str], List[str]]:
    positives: List[str] = []
    negatives: List[str] = []
    for image_path in sorted(data_root.glob("kos*/Part*.jpg")):
        stem = image_path.relative_to(data_root).with_suffix("").as_posix()
        mask_path = image_path.with_name(image_path.stem + "_label.bmp")
        if not mask_path.exists():
            continue
        if load_mask_binary(mask_path).any():
            positives.append(stem)
        else:
            negatives.append(stem)
    return positives, negatives


def build_same_split(cfg: TrainConfig) -> Dict[str, List[str]]:
    data_root = Path(cfg.data_root)
    positives, negatives = discover_samples(data_root)

    rng = np.random.default_rng(cfg.seed)
    positives = rng.permutation(positives).tolist()
    negatives = rng.permutation(negatives).tolist()

    train_pos = positives[:cfg.train_pos]
    val_pos = positives[cfg.train_pos:cfg.train_pos + cfg.val_pos]
    train_neg = negatives[:cfg.train_neg]
    val_neg = negatives[cfg.train_neg:cfg.train_neg + cfg.val_neg]

    train = train_pos + train_neg
    val = val_pos + val_neg
    rng.shuffle(train)
    rng.shuffle(val)

    return {
        "train": train,
        "val": val,
        "train_pos": train_pos,
        "train_neg": train_neg,
        "val_pos": val_pos,
        "val_neg": val_neg,
    }


class SDDDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        samples: List[str],
        processor: SegformerImageProcessor,
        image_size: Tuple[int, int],
        train: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.samples = samples
        self.processor = processor
        self.height, self.width = image_size
        self.train = train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rel = self.samples[idx]
        image_path = self.data_root / f"{rel}.jpg"
        mask_path = self.data_root / f"{rel}_label.bmp"

        image = Image.open(image_path).convert("RGB")
        mask = Image.fromarray(load_mask_binary(mask_path))

        if self.train and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        encoded = self.processor(
            images=image,
            segmentation_maps=mask,
            return_tensors="pt",
        )

        pixel_values = encoded["pixel_values"].squeeze(0)
        labels = encoded["labels"].squeeze(0).long()

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "sample_id": rel,
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch], dim=0),
        "labels": torch.stack([b["labels"] for b in batch], dim=0),
        "sample_ids": [b["sample_id"] for b in batch],
    }


def compute_sdd_metrics(
    preds: List[np.ndarray],
    gts: List[np.ndarray],
    lambda_fp: float,
) -> Dict[str, float]:
    tp = fp = fn = 0
    neg_fp_images = 0
    neg_fp_pixels = 0
    neg_total_pixels = 0
    neg_images = 0

    for pred, gt in zip(preds, gts):
        pred_pos = pred == 1
        gt_pos = gt == 1

        tp += int(np.logical_and(pred_pos, gt_pos).sum())
        fp += int(np.logical_and(pred_pos, np.logical_not(gt_pos)).sum())
        fn += int(np.logical_and(np.logical_not(pred_pos), gt_pos).sum())

        if not gt_pos.any():
            neg_images += 1
            neg_fp_images += int(pred_pos.any())
            neg_fp_pixels += int(pred_pos.sum())
            neg_total_pixels += int(gt.size)

    pos_dice = 2.0 * tp / (2.0 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    pos_iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    neg_fp_rate = neg_fp_images / neg_images if neg_images > 0 else 0.0
    neg_fp_pixel_rate = neg_fp_pixels / neg_total_pixels if neg_total_pixels > 0 else 0.0
    sdd_score = pos_dice - lambda_fp * neg_fp_rate

    return {
        "pos_dice": pos_dice,
        "pos_iou": pos_iou,
        "neg_fp_rate": neg_fp_rate,
        "neg_fp_pixel_rate": neg_fp_pixel_rate,
        "sdd_score": sdd_score,
    }


def overlay_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    image = image.astype(np.uint8)
    color = np.zeros_like(image)
    color[mask == 1] = np.array([255, 255, 255], dtype=np.uint8)
    return (0.5 * image + 0.5 * color).astype(np.uint8)


def save_previews(
    model: SegformerForSemanticSegmentation,
    processor: SegformerImageProcessor,
    samples: List[str],
    data_root: Path,
    image_size: Tuple[int, int],
    out_dir: Path,
    device: torch.device,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    height, width = image_size

    model.eval()
    for sample in samples:
        image_path = data_root / f"{sample}.jpg"
        mask_path = data_root / f"{sample}_label.bmp"
        image = Image.open(image_path).convert("RGB")
        gt_mask = Image.fromarray(load_mask_binary(mask_path))

        encoded = processor(images=image, return_tensors="pt")
        pixel_values = encoded["pixel_values"].to(device)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
        logits = outputs.logits
        logits = F.interpolate(
            logits,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        image_resized = np.array(image.resize((width, height), resample=Image.BILINEAR))
        gt_resized = np.array(gt_mask.resize((width, height), resample=Image.NEAREST)).astype(np.uint8)

        pred_overlay = overlay_mask(image_resized, pred)
        gt_overlay = overlay_mask(image_resized, gt_resized)

        stem = sample.replace("/", "__")
        Image.fromarray(image_resized).save(out_dir / f"{stem}_image.jpg")
        Image.fromarray(pred_overlay).save(out_dir / f"{stem}_pred.jpg")
        Image.fromarray(gt_overlay).save(out_dir / f"{stem}_gt.jpg")
        Image.fromarray((pred * 255).astype(np.uint8)).save(out_dir / f"{stem}_pred_mask.png")
        Image.fromarray((gt_resized * 255).astype(np.uint8)).save(out_dir / f"{stem}_gt_mask.png")


def evaluate(
    model: SegformerForSemanticSegmentation,
    loader: DataLoader,
    device: torch.device,
    lambda_fp: float,
) -> Dict[str, float]:
    model.eval()
    preds: List[np.ndarray] = []
    gts: List[np.ndarray] = []

    for batch in tqdm(loader, desc="Validating", leave=False):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].cpu().numpy().astype(np.uint8)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
        logits = outputs.logits
        logits = F.interpolate(
            logits,
            size=batch["labels"].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        pred = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)

        preds.extend(list(pred))
        gts.extend(list(labels))

    return compute_sdd_metrics(preds, gts, lambda_fp=lambda_fp)


def save_history_plot(history: List[Dict[str, float]], out_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    pos_dice = [row["pos_dice"] for row in history]
    pos_iou = [row["pos_iou"] for row in history]
    sdd_score = [row["sdd_score"] for row in history]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, pos_dice, label="pos_dice")
    plt.plot(epochs, pos_iou, label="pos_iou")
    plt.plot(epochs, sdd_score, label="sdd_score")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("SegFormer SDD Validation Metrics")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def train(cfg: TrainConfig) -> None:
    set_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "previews").mkdir(exist_ok=True)

    splits = build_same_split(cfg)
    with open(output_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_positive": len(splits["train_pos"]),
                "train_negative": len(splits["train_neg"]),
                "val_positive": len(splits["val_pos"]),
                "val_negative": len(splits["val_neg"]),
                "train_samples": splits["train"],
                "val_samples": splits["val"],
            },
            f,
            indent=2,
        )

    processor = SegformerImageProcessor(
        do_resize=True,
        size={"height": cfg.image_height, "width": cfg.image_width},
        do_reduce_labels=False,
    )

    train_ds = SDDDataset(
        data_root=cfg.data_root,
        samples=splits["train"],
        processor=processor,
        image_size=(cfg.image_height, cfg.image_width),
        train=True,
    )
    val_ds = SDDDataset(
        data_root=cfg.data_root,
        samples=splits["val"],
        processor=processor,
        image_size=(cfg.image_height, cfg.image_width),
        train=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegformerForSemanticSegmentation.from_pretrained(
        cfg.model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    history: List[Dict[str, float]] = []
    best_score = -float("inf")

    val_preview_samples = (
        splits["val_pos"][:1] + splits["val_neg"][:1]
    )[:cfg.preview_samples]

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        prog = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False)
        for batch in prog:
            optimizer.zero_grad(set_to_none=True)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            prog.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(len(train_loader), 1)
        metrics = evaluate(model, val_loader, device=device, lambda_fp=cfg.lambda_fp)
        metrics["epoch"] = epoch
        metrics["train_loss"] = train_loss
        history.append(metrics)

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"pos_dice={metrics['pos_dice']:.4f}, "
            f"pos_iou={metrics['pos_iou']:.4f}, "
            f"neg_fp_rate={metrics['neg_fp_rate']:.4f}, "
            f"neg_fp_pixel_rate={metrics['neg_fp_pixel_rate']:.4f}, "
            f"sdd_score={metrics['sdd_score']:.4f}"
        )

        save_previews(
            model=model,
            processor=processor,
            samples=val_preview_samples,
            data_root=Path(cfg.data_root),
            image_size=(cfg.image_height, cfg.image_width),
            out_dir=output_dir / "previews" / f"epoch_{epoch:03d}",
            device=device,
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "config": asdict(cfg),
            },
            output_dir / "latest.pt",
        )

        if metrics["sdd_score"] > best_score:
            best_score = metrics["sdd_score"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": metrics,
                    "config": asdict(cfg),
                },
                output_dir / "best_sdd_score.pt",
            )

        with open(output_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        save_history_plot(history, output_dir / "history.png")

    print("Training complete.")
    print(f"Best sdd_score: {best_score:.6f}")
    print(f"Outputs saved to: {output_dir}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train SegFormer on Kolektor SDD for Colab.")
    parser.add_argument("--data-root", required=True, help="Path to the SDD root folder")
    parser.add_argument("--output-dir", required=True, help="Where to save checkpoints and previews")
    parser.add_argument("--model-name", default="nvidia/segformer-b1-finetuned-ade-512-512")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-height", type=int, default=512)
    parser.add_argument("--image-width", type=int, default=1408)
    parser.add_argument("--train-pos", type=int, default=41)
    parser.add_argument("--train-neg", type=int, default=82)
    parser.add_argument("--val-pos", type=int, default=11)
    parser.add_argument("--val-neg", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--preview-samples", type=int, default=2)
    parser.add_argument("--lambda-fp", type=float, default=0.10)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
