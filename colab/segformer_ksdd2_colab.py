"""
Colab-ready SegFormer training script for KolektorSDD2 / KSDD2.

Why this version:
- Uses the modern Hugging Face Transformers SegFormer implementation.
- Targets Google Colab / GPU directly instead of the legacy MMSeg stack.
- Preserves the same evaluation logic we used locally:
  * pos_dice
  * pos_iou
  * neg_fp_rate
  * neg_fp_pixel_rate
  * sdd_score = pos_dice - lambda_fp * neg_fp_rate

KSDD2 sizing choice:
- The official KSDD2 page describes images as approximately 230 x 630 pixels.
- This script resizes each image to fit inside that box while preserving aspect ratio,
  then pads to a common model input size so every batch has identical shape.
- By default it pads to multiples of 32, which gives 256 x 640 from 230 x 630.

Primary sources:
- https://www.vicos.si/resources/kolektorsdd2/
- https://huggingface.co/docs/transformers/model_doc/segformer

Suggested Colab install cell:
!pip install -q transformers accelerate torch torchvision pillow matplotlib tqdm

Example run:
!python segformer_ksdd2_colab.py \
    --data-root /content/KolektorSDD2 \
    --output-dir /content/segformer_ksdd2_runs \
    --epochs 20 \
    --batch-size 8 \
    --model-name nvidia/segformer-b1-finetuned-ade-512-512
"""

from __future__ import annotations

import argparse
import json
import math
import random
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


ID2LABEL = {0: "background", 1: "positive"}
LABEL2ID = {"background": 0, "positive": 1}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_HINTS = {"mask", "masks", "gt", "ground_truth", "label", "labels"}


@dataclass
class TrainConfig:
    data_root: str
    output_dir: str
    model_name: str = "nvidia/segformer-b1-finetuned-ade-512-512"
    seed: int = 42
    resize_height: int = 230
    resize_width: int = 630
    pad_multiple: int = 32
    batch_size: int = 8
    eval_batch_size: int = 4
    epochs: int = 20
    lr: float = 6e-5
    weight_decay: float = 1e-2
    preview_samples: int = 2
    lambda_fp: float = 0.10
    num_workers: int = 2


@dataclass
class Sample:
    image_path: str
    mask_path: Optional[str]
    split: str
    sample_id: str
    label: int


_TEMP_DATASETS: List[tempfile.TemporaryDirectory] = []


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def round_up(value: int, multiple: int) -> int:
    if multiple <= 1:
        return value
    return int(math.ceil(value / multiple) * multiple)


def load_mask_binary(mask_path: Path, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    if not mask_path.exists():
        if size is None:
            raise FileNotFoundError(mask_path)
        return np.zeros(size, dtype=np.uint8)
    mask = Image.open(mask_path)
    mask_arr = np.array(mask)
    return (mask_arr > 0).astype(np.uint8)


def is_mask_path(path: Path) -> bool:
    lower = path.stem.lower()
    if any(token in lower for token in ("_label", "_mask", "_gt")):
        return True
    if any(part.lower() in MASK_HINTS for part in path.parts):
        return True
    return False


def candidate_mask_paths(image_path: Path) -> List[Path]:
    stems = [
        f"{image_path.stem}_GT",
        f"{image_path.stem}_label",
        f"{image_path.stem}_mask",
        f"{image_path.stem}_gt",
        image_path.stem,
    ]
    suffixes = [".png", ".bmp", ".jpg", ".jpeg", ".tif", ".tiff"]
    candidates: List[Path] = []

    for stem in stems:
        for suffix in suffixes:
            candidates.append(image_path.with_name(stem + suffix))

    parent = image_path.parent
    split_root = parent.parent if parent.name.lower() in {"ok", "defect"} else parent
    for folder in ("masks", "masks_defect", "ground_truth", "gt", "label", "labels"):
        for suffix in suffixes:
            candidates.append(split_root / folder / f"{image_path.stem}{suffix}")
            candidates.append(split_root / folder / f"{image_path.stem}_label{suffix}")
            candidates.append(split_root / folder / f"{image_path.stem}_mask{suffix}")
            candidates.append(split_root / folder / f"{image_path.stem}_gt{suffix}")
    return candidates


def resolve_mask_path(image_path: Path) -> Optional[Path]:
    for candidate in candidate_mask_paths(image_path):
        if candidate.exists() and candidate.resolve() != image_path.resolve():
            return candidate
    return None


def infer_label(image_path: Path, mask_path: Optional[Path]) -> int:
    parts = {part.lower() for part in image_path.parts}
    if "defect" in parts:
        return 1
    if "ok" in parts:
        return 0
    if mask_path is None:
        return 0
    with Image.open(image_path) as image:
        mask = load_mask_binary(mask_path, size=(image.height, image.width))
    return int(mask.any())


def collect_split_samples(split_dir: Path, split_name: str) -> List[Sample]:
    samples: List[Sample] = []
    if not split_dir.exists():
        return samples

    for image_path in sorted(split_dir.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        if "(copy)" in image_path.name.lower():
            continue
        if is_mask_path(image_path):
            continue

        mask_path = resolve_mask_path(image_path)
        label = infer_label(image_path, mask_path)
        sample_id = image_path.relative_to(split_dir).with_suffix("").as_posix()
        samples.append(
            Sample(
                image_path=str(image_path),
                mask_path=str(mask_path) if mask_path is not None else None,
                split=split_name,
                sample_id=sample_id,
                label=label,
            )
        )
    return samples


def collect_flat_samples(data_root: Path) -> List[Sample]:
    return collect_split_samples(data_root, "all")


def find_child_dir_case_insensitive(parent: Path, name: str) -> Optional[Path]:
    if not parent.exists():
        return None
    wanted = name.lower()
    for child in parent.iterdir():
        if child.is_dir() and child.name.lower() == wanted:
            return child
    return None


def discover_ksdd2_samples(data_root: Path) -> Dict[str, List[Sample]]:
    data_root = prepare_data_root(data_root)
    data_root = resolve_data_root(data_root)
    train_dir = find_child_dir_case_insensitive(data_root, "train") or (data_root / "train")
    test_dir = find_child_dir_case_insensitive(data_root, "test") or (data_root / "test")
    train_samples = collect_split_samples(train_dir, "train")
    test_samples = collect_split_samples(test_dir, "test")

    if not train_samples and not test_samples:
        flat_samples = collect_flat_samples(data_root)
        if flat_samples:
            rng = random.Random(42)
            positives = [sample for sample in flat_samples if sample.label == 1]
            negatives = [sample for sample in flat_samples if sample.label == 0]
            rng.shuffle(positives)
            rng.shuffle(negatives)
            train_samples = positives[:246] + negatives[:2085]
            test_samples = positives[246:356] + negatives[2085:2979]
            rng.shuffle(train_samples)
            rng.shuffle(test_samples)
            print(
                "No train/test files found, so using official KSDD2 counts "
                "from a flat folder: train=246 positive/2085 negative, "
                "test=110 positive/894 negative."
            )

    if not train_samples or not test_samples:
        existing_dirs = [
            str(path)
            for path in sorted(data_root.rglob("*"))
            if path.is_dir()
        ][:80]
        existing_files = [
            str(path)
            for path in sorted(data_root.rglob("*"))
            if path.is_file()
        ][:80]
        raise RuntimeError(
            "Could not find a usable KSDD2 train/test layout under data-root. "
            "Expected folders like train/ and test/ with images, optionally with "
            "ok/, defect/, masks/, gt/, or label/ subfolders.\n"
            f"Resolved data-root: {data_root}\n"
            f"Train dir checked: {train_dir} ({len(train_samples)} samples)\n"
            f"Test dir checked: {test_dir} ({len(test_samples)} samples)\n"
            "First directories found under data-root:\n"
            + "\n".join(existing_dirs)
            + "\nFirst files found under data-root:\n"
            + "\n".join(existing_files)
        )

    return {"train": train_samples, "test": test_samples}


def prepare_data_root(data_root: Path) -> Path:
    if data_root.is_file() and data_root.suffix.lower() == ".zip":
        tmp = tempfile.TemporaryDirectory(prefix="ksdd2_")
        _TEMP_DATASETS.append(tmp)
        out_dir = Path(tmp.name)
        print(f"Extracting KSDD2 zip to temporary folder: {out_dir}")
        with zipfile.ZipFile(data_root, "r") as archive:
            archive.extractall(out_dir)
        return out_dir
    return data_root


def resolve_data_root(data_root: Path) -> Path:
    if (
        find_child_dir_case_insensitive(data_root, "train") is not None
        and find_child_dir_case_insensitive(data_root, "test") is not None
    ):
        return data_root

    matches = [
        path
        for path in data_root.rglob("*")
        if (
            path.is_dir()
            and find_child_dir_case_insensitive(path, "train") is not None
            and find_child_dir_case_insensitive(path, "test") is not None
        )
    ]
    if matches:
        resolved = sorted(matches, key=lambda path: len(path.parts))[0]
        print(f"Resolved nested KSDD2 root: {resolved}")
        return resolved

    return data_root


def resize_and_pad(
    image: Image.Image,
    mask: Image.Image,
    resize_height: int,
    resize_width: int,
    pad_multiple: int,
) -> Tuple[Image.Image, Image.Image]:
    target_h = resize_height
    target_w = resize_width
    final_h = round_up(target_h, pad_multiple)
    final_w = round_up(target_w, pad_multiple)

    src_w, src_h = image.size
    scale = min(target_w / src_w, target_h / src_h)
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    image_resized = image.resize((new_w, new_h), resample=Image.BILINEAR)
    mask_resized = mask.resize((new_w, new_h), resample=Image.NEAREST)

    pad_left = (final_w - new_w) // 2
    pad_top = (final_h - new_h) // 2
    pad_right = final_w - new_w - pad_left
    pad_bottom = final_h - new_h - pad_top
    padding = (pad_left, pad_top, pad_right, pad_bottom)

    image_padded = ImageOps.expand(image_resized, border=padding, fill=0)
    mask_padded = ImageOps.expand(mask_resized, border=padding, fill=0)
    return image_padded, mask_padded


class KSDD2Dataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        processor: SegformerImageProcessor,
        resize_height: int,
        resize_width: int,
        pad_multiple: int,
        train: bool = False,
    ) -> None:
        self.samples = list(samples)
        self.processor = processor
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.pad_multiple = pad_multiple
        self.train = train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        image_path = Path(sample.image_path)
        image = Image.open(image_path).convert("RGB")
        if sample.mask_path is None:
            mask_arr = np.zeros((image.height, image.width), dtype=np.uint8)
        else:
            mask_arr = load_mask_binary(Path(sample.mask_path), size=(image.height, image.width))
        mask = Image.fromarray(mask_arr)

        if self.train and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        image, mask = resize_and_pad(
            image=image,
            mask=mask,
            resize_height=self.resize_height,
            resize_width=self.resize_width,
            pad_multiple=self.pad_multiple,
        )

        encoded = self.processor(
            images=image,
            segmentation_maps=mask,
            return_tensors="pt",
        )

        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels": encoded["labels"].squeeze(0).long(),
            "sample_id": sample.sample_id,
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch], dim=0),
        "labels": torch.stack([item["labels"] for item in batch], dim=0),
        "sample_ids": [item["sample_id"] for item in batch],
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
    color = np.zeros_like(image, dtype=np.uint8)
    color[mask == 1] = np.array([255, 255, 255], dtype=np.uint8)
    return (0.5 * image.astype(np.float32) + 0.5 * color.astype(np.float32)).astype(np.uint8)


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
        logits = F.interpolate(
            outputs.logits,
            size=batch["labels"].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        pred = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
        preds.extend(list(pred))
        gts.extend(list(labels))

    return compute_sdd_metrics(preds, gts, lambda_fp=lambda_fp)


def save_previews(
    model: SegformerForSemanticSegmentation,
    processor: SegformerImageProcessor,
    samples: Sequence[Sample],
    out_dir: Path,
    cfg: TrainConfig,
    device: torch.device,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    for sample in samples:
        image = Image.open(sample.image_path).convert("RGB")
        if sample.mask_path is None:
            gt_mask = Image.fromarray(np.zeros((image.height, image.width), dtype=np.uint8))
        else:
            gt_mask = Image.fromarray(load_mask_binary(Path(sample.mask_path), size=(image.height, image.width)))

        image_padded, gt_padded = resize_and_pad(
            image=image,
            mask=gt_mask,
            resize_height=cfg.resize_height,
            resize_width=cfg.resize_width,
            pad_multiple=cfg.pad_multiple,
        )

        encoded = processor(images=image_padded, return_tensors="pt")
        pixel_values = encoded["pixel_values"].to(device)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
        logits = F.interpolate(
            outputs.logits,
            size=(gt_padded.size[1], gt_padded.size[0]),
            mode="bilinear",
            align_corners=False,
        )
        pred = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        image_np = np.array(image_padded)
        gt_np = np.array(gt_padded).astype(np.uint8)

        stem = sample.sample_id.replace("/", "__")
        Image.fromarray(image_np).save(out_dir / f"{stem}_image.jpg")
        Image.fromarray(overlay_mask(image_np, pred)).save(out_dir / f"{stem}_pred.jpg")
        Image.fromarray(overlay_mask(image_np, gt_np)).save(out_dir / f"{stem}_gt.jpg")
        Image.fromarray((pred * 255).astype(np.uint8)).save(out_dir / f"{stem}_pred_mask.png")
        Image.fromarray((gt_np * 255).astype(np.uint8)).save(out_dir / f"{stem}_gt_mask.png")


def save_history_plot(history: List[Dict[str, float]], out_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [row["pos_dice"] for row in history], label="pos_dice")
    plt.plot(epochs, [row["pos_iou"] for row in history], label="pos_iou")
    plt.plot(epochs, [row["sdd_score"] for row in history], label="sdd_score")
    plt.xlabel("Epoch")
    plt.ylabel("Metric")
    plt.title("SegFormer KSDD2 Validation Metrics")
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

    split_samples = discover_ksdd2_samples(Path(cfg.data_root))
    train_samples = split_samples["train"]
    test_samples = split_samples["test"]

    with open(output_dir / "split_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "source": "KSDD2 official train/test split",
                "train_total": len(train_samples),
                "train_positive": sum(sample.label for sample in train_samples),
                "train_negative": len(train_samples) - sum(sample.label for sample in train_samples),
                "test_total": len(test_samples),
                "test_positive": sum(sample.label for sample in test_samples),
                "test_negative": len(test_samples) - sum(sample.label for sample in test_samples),
                "resize_box": [cfg.resize_height, cfg.resize_width],
                "pad_multiple": cfg.pad_multiple,
                "padded_shape": [
                    round_up(cfg.resize_height, cfg.pad_multiple),
                    round_up(cfg.resize_width, cfg.pad_multiple),
                ],
            },
            handle,
            indent=2,
        )

    processor = SegformerImageProcessor(
        do_resize=False,
        do_reduce_labels=False,
    )

    train_ds = KSDD2Dataset(
        samples=train_samples,
        processor=processor,
        resize_height=cfg.resize_height,
        resize_width=cfg.resize_width,
        pad_multiple=cfg.pad_multiple,
        train=True,
    )
    test_ds = KSDD2Dataset(
        samples=test_samples,
        processor=processor,
        resize_height=cfg.resize_height,
        resize_width=cfg.resize_width,
        pad_multiple=cfg.pad_multiple,
        train=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
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

    best_score = -float("inf")
    history: List[Dict[str, float]] = []
    preview_samples = test_samples[:cfg.preview_samples]

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.epochs}", leave=False)

        for batch in train_bar:
            optimizer.zero_grad(set_to_none=True)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running_loss / max(len(train_loader), 1)
        metrics = evaluate(model, test_loader, device=device, lambda_fp=cfg.lambda_fp)
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
            samples=preview_samples,
            out_dir=output_dir / "previews" / f"epoch_{epoch:03d}",
            cfg=cfg,
            device=device,
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": asdict(cfg),
        }
        torch.save(checkpoint, output_dir / "latest.pt")
        if metrics["sdd_score"] > best_score:
            best_score = metrics["sdd_score"]
            torch.save(checkpoint, output_dir / "best_sdd_score.pt")

        with open(output_dir / "history.json", "w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)
        save_history_plot(history, output_dir / "history.png")

    print("Training complete.")
    print(f"Best sdd_score: {best_score:.6f}")
    print(f"Outputs saved to: {output_dir}")


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train SegFormer on KSDD2 for Colab.")
    parser.add_argument("--data-root", required=True, help="Path to the KSDD2 root folder")
    parser.add_argument("--output-dir", required=True, help="Where checkpoints and previews are saved")
    parser.add_argument("--model-name", default="nvidia/segformer-b1-finetuned-ade-512-512")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resize-height", type=int, default=230)
    parser.add_argument("--resize-width", type=int, default=630)
    parser.add_argument("--pad-multiple", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--preview-samples", type=int, default=2)
    parser.add_argument("--lambda-fp", type=float, default=0.10)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    train(parse_args())
