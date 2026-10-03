#!/usr/bin/env python3
"""运行 SIGNet 从结构分割到病例分级的完整推理流程。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Deidentified PNG directory")
    parser.add_argument("--slice-manifest", required=True)
    parser.add_argument("--segmentation-checkpoints", required=True)
    parser.add_argument("--grading-checkpoint-left", required=True)
    parser.add_argument("--grading-checkpoint-right", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--thresholds", default=None, help="Optional validation-selected overrides")
    parser.add_argument("--minimum-consecutive", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.minimum_consecutive != 2:
        raise ValueError("--minimum-consecutive must be 2 for this model configuration")

    # 定义第一阶段分割结果和第二阶段分级结果的输出路径。
    output_root = Path(args.output_dir)
    segmentation_output = output_root / "segment"
    grading_output = output_root / "grading"

    # 第一阶段：分割 LI、LS、RI、RS，并生成左右关节 ROI 和预测掩膜。
    segment_command = [
        sys.executable,
        "-m",
        "segment.predict",
        "--input-dir",
        args.input_dir,
        "--checkpoint-dir",
        args.segmentation_checkpoints,
        "--output-dir",
        str(segmentation_output),
    ]
    if args.thresholds:
        segment_command.extend(["--thresholds", args.thresholds])
    subprocess.run(segment_command, check=True)

    # 第二阶段：融合灰度 ROI 与三通道结构先验，输出切片级和病例级分级。
    subprocess.run(
        [
            sys.executable,
            "-m",
            "model.predict",
            "--roi-dir",
            str(segmentation_output / "roi"),
            "--mask-dir",
            str(segmentation_output / "roi_masks"),
            "--slice-manifest",
            args.slice_manifest,
            "--checkpoint-left",
            args.grading_checkpoint_left,
            "--checkpoint-right",
            args.grading_checkpoint_right,
            "--output-dir",
            str(grading_output),
            "--minimum-consecutive",
            str(args.minimum_consecutive),
        ],
        check=True,
    )
    print(f"SIGNet outputs are available in {output_root}")


if __name__ == "__main__":
    main()
