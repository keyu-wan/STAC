#!/usr/bin/env python3
"""Convert nnU-Net .npz probability exports to STAC float32 maps and hard masks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--foreground-class", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def foreground_probability(
    probabilities: np.ndarray, class_index: int, case_id: str
) -> np.ndarray:
    if probabilities.ndim == 4:
        if probabilities.shape[1] != 1:
            raise ValueError(
                f"Expected singleton depth for 2D predictions ({case_id}), "
                f"got {probabilities.shape}"
            )
        probabilities = probabilities[:, 0]
    if probabilities.ndim != 3:
        raise ValueError(
            f"Expected [classes, height, width] probabilities for {case_id}, "
            f"got {probabilities.shape}"
        )
    if not 0 <= class_index < probabilities.shape[0]:
        raise ValueError(
            f"Foreground class {class_index} is out of range for {case_id}"
        )
    result = np.asarray(probabilities[class_index], dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError(f"Non-finite probabilities for {case_id}")
    if np.any((result < -1e-6) | (result > 1.0 + 1e-6)):
        raise ValueError(f"Probabilities outside [0, 1] for {case_id}")
    return np.clip(result, 0.0, 1.0)


def main() -> None:
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("--threshold must be in (0, 1)")
    if not args.prediction_dir.is_dir():
        raise FileNotFoundError(args.prediction_dir)
    archives = sorted(args.prediction_dir.glob("*.npz"))
    if not archives:
        raise FileNotFoundError(f"No .npz predictions in {args.prediction_dir}")

    probability_dir = args.output_dir / "probabilities"
    hard_mask_dir = args.output_dir / "hard_masks"
    probability_dir.mkdir(parents=True, exist_ok=True)
    hard_mask_dir.mkdir(parents=True, exist_ok=True)

    for index, archive in enumerate(archives, start=1):
        probability_path = probability_dir / f"{archive.stem}.npy"
        hard_mask_path = hard_mask_dir / f"{archive.stem}.png"
        if probability_path.exists() or hard_mask_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite existing artifacts for {archive.stem}"
            )
        with np.load(archive, allow_pickle=False) as payload:
            if "probabilities" not in payload:
                raise ValueError(f"Missing 'probabilities' array in {archive}")
            probability = foreground_probability(
                payload["probabilities"], args.foreground_class, archive.stem
            )
        hard = probability > args.threshold
        np.save(probability_path, probability.astype(np.float32, copy=False))
        Image.fromarray(hard.astype(np.uint8) * 255, mode="L").save(hard_mask_path)
        if index % 100 == 0 or index == len(archives):
            print(f"converted: {index}/{len(archives)}", flush=True)

    print(
        f"Wrote {len(archives)} cases under {args.output_dir.resolve()} "
        f"(class={args.foreground_class}, threshold={args.threshold})"
    )


if __name__ == "__main__":
    main()
