#!/usr/bin/env python3
"""Train the four independent SIGNet segmentation models."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from segment.dataset import STRUCTURES, StructureDataset
from segment.losses import SegmentationObjective
from segment.model import UnetVMamba, main_logits
from utils import environment_metadata, seed_everything, sha256_file, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, help="Contains train/ and val/")
    parser.add_argument("--output-dir", default="outputs/segmentation")
    parser.add_argument("--structures", nargs="+", default=list(STRUCTURES))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def accumulation_divisor(step: int, total_steps: int, accumulation_steps: int) -> int:
    """Return the actual number of micro-batches in the current update group."""
    if accumulation_steps < 1:
        raise ValueError("gradient accumulation must be positive")
    if not 1 <= step <= total_steps:
        raise ValueError("step must be within the current epoch")
    remainder = total_steps % accumulation_steps
    if remainder and step > total_steps - remainder:
        return remainder
    return accumulation_steps


@torch.no_grad()
def validate(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[float, float, float, float]:
    model.eval()
    probabilities, targets = [], []
    for batch in loader:
        image = batch["image"].to(device)
        logits = main_logits(model(image))
        probabilities.append(torch.softmax(logits, dim=1)[:, 1].cpu())
        targets.append(batch["target"].cpu())
    if not probabilities:
        raise ValueError("The validation split contains no usable samples")
    probability, target = torch.cat(probabilities), torch.cat(targets).bool()
    best = (0.5, -1.0, -1.0, -1.0)
    # Match the experiment script: search 0.01-0.99 on the frozen validation set.
    for index in range(1, 100):
        threshold = index / 100
        prediction = probability > threshold
        intersection = (prediction & target).sum(dim=(1, 2)).float()
        dice = (2 * intersection / (prediction.sum((1, 2)) + target.sum((1, 2)) + 1e-6)).mean().item()
        union = (prediction | target).sum(dim=(1, 2)).float()
        iou = (intersection / (union + 1e-6)).mean().item()
        accuracy = (prediction == target).float().mean(dim=(1, 2)).mean().item()
        if dice > best[1]:
            best = (float(threshold), dice, iou, accuracy)
    return best


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    structure: str,
    epoch: int,
    row: dict[str, float | int],
) -> None:
    """Save the anatomical structure name and validation metadata."""
    torch.save(
        {
            "format_version": 1,
            "model": model.state_dict(),
            "structure": structure,
            "side": STRUCTURES[structure][0],
            "epoch": int(epoch),
            "threshold": float(row["validation_threshold"]),
            "threshold_source": "Dataset A validation set",
            "threshold_search": {"start": 0.01, "stop": 0.99, "step": 0.01},
            "validation_metrics": {
                "dice": float(row["validation_dice"]),
                "iou": float(row["validation_iou"]),
                "accuracy": float(row["validation_accuracy"]),
            },
            "model_config": {
                "in_channels": 1,
                "num_classes": 2,
                "base_channels": 48,
                "drop_path_rate": 0.10,
            },
        },
        path,
    )


def train_structure(
    args: argparse.Namespace, structure: str, device: torch.device
) -> dict[str, str | float | int]:
    # 为 LI、LS、RI、RS 中的当前结构分别创建训练集和验证集。
    train_set = StructureDataset(Path(args.data_root) / "train", structure, train=True)
    val_set = StructureDataset(Path(args.data_root) / "val", structure, train=False)
    train_loader = DataLoader(
        train_set,
        args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_set, args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    if not train_set:
        raise ValueError(f"The training split contains no usable samples for {structure}")
    model = UnetVMamba(in_channels=1, num_classes=2, base_channels=48, drop_path_rate=0.10).to(device)
    # The objective owns registered buffers (class weights and Sobel kernels),
    # so it must follow the model and inputs onto the selected device.
    objective = SegmentationObjective().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.1
    )
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best: dict[int, tuple[float, int, float]] = {
        10: (-1.0, 0, 0.5),
        20: (-1.0, 0, 0.5),
        30: (-1.0, 0, 0.5),
    }
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            tensors = {key: value.to(device) for key, value in batch.items() if torch.is_tensor(value)}
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                loss = objective(
                    model, tensors["image"], tensors["target"], tensors["rectangle"],
                    tensors["rect_image"], tensors["rect_target"], tensors["rect_rectangle"],
                    tensors["roi_image"], tensors["roi_target"], tensors["roi_rectangle"],
                )
                # The final update group may contain fewer than four
                # micro-batches. Divide by its true size so it is not
                # underweighted relative to complete update groups.
                divisor = accumulation_divisor(step, len(train_loader), args.gradient_accumulation)
                scaled_loss = loss / divisor
            scaler.scale(scaled_loss).backward()
            running += float(loss.item())
            # 累积指定数量的小批次后再更新参数。
            if step % args.gradient_accumulation == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        threshold, dice, iou, accuracy = validate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "loss": running / len(train_loader),
            "validation_threshold": threshold,
            "validation_dice": dice,
            "validation_iou": iou,
            "validation_accuracy": accuracy,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(
            f"[{structure}] epoch={epoch:03d} loss={row['loss']:.4f} "
            f"dice={dice:.4f} iou={iou:.4f} threshold={threshold:.2f}"
        )
        for horizon in best:
            if epoch <= horizon and dice > best[horizon][0]:
                best[horizon] = (dice, epoch, threshold)
                save_checkpoint(
                    output_dir / f"unet_vmamba_best{horizon}_{structure}.pt",
                    model,
                    structure,
                    epoch,
                    row,
                )

    final_row = history[-1]
    save_checkpoint(
        output_dir / f"unet_vmamba_final_{structure}.pt",
        model,
        structure,
        args.epochs,
        final_row,
    )
    log_path = output_dir / f"train_log_{structure}.json"
    log_path.write_text(
        json.dumps(
            {
                "structure": structure,
                "side": STRUCTURES[structure][0],
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    best30_path = output_dir / f"unet_vmamba_best30_{structure}.pt"
    write_json(
        output_dir / f"metadata_{structure}.json",
        {
            "environment": environment_metadata(),
            "arguments": vars(args),
            "structure": structure,
            "train_samples": len(train_set),
            "validation_samples": len(val_set),
            "best30_checkpoint_sha256": sha256_file(best30_path),
            "best_validation_dice": best[30][0],
        },
    )
    return {
        "structure": structure,
        "side": STRUCTURES[structure][0],
        "best10_dice": best[10][0],
        "best20_dice": best[20][0],
        "best30_dice": best[30][0],
        "final_threshold": final_row["validation_threshold"],
        "final_dice": final_row["validation_dice"],
        "final_iou": final_row["validation_iou"],
        "final_accuracy": final_row["validation_accuracy"],
    }


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.gradient_accumulation < 1:
        raise ValueError("epochs, batch-size and gradient-accumulation must be positive")
    if args.workers < 0:
        raise ValueError("workers cannot be negative")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summaries: list[dict[str, str | float | int]] = []
    for structure in args.structures:
        structure = structure.upper()
        if structure not in STRUCTURES:
            raise ValueError(f"Unknown structure {structure}; choose from {sorted(STRUCTURES)}")
        # 每个结构训练前重置随机种子，避免训练顺序影响结果。
        seed_everything(args.seed)
        summaries.append(train_structure(args, structure, device))

    output_dir = Path(args.output_dir)
    with (output_dir / "summary_selected_512x256.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


if __name__ == "__main__":
    main()
