#!/usr/bin/env python3
"""Build a select-method-v3 feature CSV from generic segmentation artifacts.

Inputs are image, label, probability, and hard-mask directories. The script is
model-agnostic: any warm-up model can be used as long as it exports one
float32 probability map and one thresholded hard mask per candidate case.

Group metadata is optional. If no group option is provided, the output CSV does
not contain a group column and the selector will use its built-in fallback.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
from scipy import ndimage

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a generic v3 feature CSV from segmentation artifacts."
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--probability-dir", type=Path, required=True)
    parser.add_argument("--hard-mask-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--case-ids",
        type=Path,
        help=(
            "Optional case list. Supports JSON list, JSON object with case_ids/cases, "
            "or newline-delimited text. Defaults to probability .npy stems."
        ),
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        help="Optional CSV containing case-to-group metadata.",
    )
    parser.add_argument("--metadata-case-column", default="case_id")
    parser.add_argument(
        "--metadata-group-column",
        default="group",
        help="Column in --metadata-csv to copy into the output group column.",
    )
    parser.add_argument(
        "--group-from-prefix",
        action="store_true",
        help="Set group to the substring before the first underscore in case_id.",
    )
    parser.add_argument(
        "--group-regex",
        help=(
            "Regex applied to case_id. Uses named group 'group' if present; "
            "otherwise the first capture group."
        ),
    )
    parser.add_argument(
        "--group-value",
        help="Constant group value for all rows.",
    )
    return parser.parse_args()


def load_case_ids(path: Path | None, probability_dir: Path) -> list[str]:
    if path is None:
        case_ids = sorted(p.stem for p in probability_dir.glob("*.npy"))
    else:
        text = path.read_text(encoding="utf-8")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = [line.strip() for line in text.splitlines() if line.strip()]
        if isinstance(payload, dict):
            values = payload.get("case_ids", payload.get("cases"))
        else:
            values = payload
        if not isinstance(values, list):
            raise ValueError(f"Unsupported case-id file format: {path}")
        case_ids = [Path(str(value)).stem for value in values if str(value).strip()]
    if not case_ids:
        raise ValueError("No case IDs found")
    if len(case_ids) != len(set(case_ids)):
        duplicates = sorted({case for case in case_ids if case_ids.count(case) > 1})
        raise ValueError(f"Duplicate case IDs: {duplicates[:10]}")
    return case_ids


def index_by_stem(
    directory: Path,
    extensions: Iterable[str] | None = None,
    *,
    strip_nnunet_channel: bool = False,
) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    allowed = None if extensions is None else {ext.lower() for ext in extensions}
    indexed: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if path.name.startswith(".") or not path.is_file():
            continue
        if allowed is not None and path.suffix.lower() not in allowed:
            continue
        stem = path.stem
        if strip_nnunet_channel and stem.endswith("_0000"):
            stem = stem[:-5]
        if stem in indexed:
            raise ValueError(f"Duplicate stem {stem!r} in {directory}")
        indexed[stem] = path
    return indexed


def load_metadata_groups(args: argparse.Namespace) -> dict[str, str]:
    if args.metadata_csv is None:
        return {}
    groups: dict[str, str] = {}
    with args.metadata_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if args.metadata_case_column not in (reader.fieldnames or []):
            raise ValueError(
                f"metadata case column {args.metadata_case_column!r} not found in {args.metadata_csv}"
            )
        if args.metadata_group_column not in (reader.fieldnames or []):
            raise ValueError(
                f"metadata group column {args.metadata_group_column!r} not found in {args.metadata_csv}"
            )
        for row in reader:
            case_id = Path(str(row[args.metadata_case_column]).strip()).stem
            group = str(row[args.metadata_group_column]).strip()
            if case_id and group:
                groups[case_id] = group
    return groups


def build_group_resolver(args: argparse.Namespace):
    modes = sum(
        bool(value)
        for value in (
            args.metadata_csv,
            args.group_from_prefix,
            args.group_regex,
            args.group_value,
        )
    )
    if modes > 1:
        raise ValueError(
            "Use at most one group source: --metadata-csv, --group-from-prefix, "
            "--group-regex, or --group-value"
        )
    if args.metadata_csv:
        groups = load_metadata_groups(args)

        def resolve(case_id: str) -> str | None:
            if case_id not in groups:
                raise KeyError(f"No group metadata for case {case_id!r}")
            return groups[case_id]

        return resolve
    if args.group_from_prefix:
        return lambda case_id: case_id.split("_", 1)[0]
    if args.group_regex:
        pattern = re.compile(args.group_regex)

        def resolve(case_id: str) -> str | None:
            match = pattern.search(case_id)
            if match is None:
                raise ValueError(f"group regex did not match case_id {case_id!r}")
            if "group" in match.groupdict():
                return match.group("group")
            if match.groups():
                return match.group(1)
            raise ValueError("--group-regex must define a capture group")

        return resolve
    if args.group_value:
        return lambda case_id: args.group_value
    return lambda case_id: None


def boundary_band(mask: np.ndarray) -> np.ndarray:
    foreground = mask.astype(bool)
    dilated = ndimage.binary_dilation(foreground, iterations=1)
    eroded = ndimage.binary_erosion(foreground, iterations=1, border_value=0)
    return np.logical_and(dilated, np.logical_not(eroded))


def compute_row(
    case_id: str,
    image_path: Path,
    label_path: Path,
    probability_path: Path,
    hard_mask_path: Path,
    threshold: float,
    group: str | None,
) -> dict[str, object]:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    gt = np.asarray(Image.open(label_path).convert("L"), dtype=np.uint8) > 0
    probability = np.load(probability_path, allow_pickle=False)
    if probability.ndim != 2:
        raise ValueError(f"Expected a 2D probability array for {case_id}")
    if probability.shape != gt.shape or image.shape[:2] != gt.shape:
        raise ValueError(
            f"Shape mismatch for {case_id}: image={image.shape[:2]}, "
            f"gt={gt.shape}, probability={probability.shape}"
        )
    if probability.dtype != np.float32:
        raise ValueError(
            f"Probability dtype must remain float32 for {case_id}, got {probability.dtype}"
        )
    if not np.isfinite(probability).all():
        raise ValueError(f"Non-finite probability values for {case_id}")
    probability = np.clip(probability, 0.0, 1.0)
    hard = probability > threshold
    exported_hard = np.asarray(Image.open(hard_mask_path).convert("L"), dtype=np.uint8) > 0
    if not np.array_equal(hard, exported_hard):
        raise ValueError(f"Hard mask does not match probability threshold for {case_id}")

    background = ~gt
    boundary = boundary_band(gt)
    hard_boundary = boundary_band(hard)
    entropy_probability = np.clip(probability, 1e-6, 1.0 - 1e-6)
    entropy = -(
        entropy_probability * np.log(entropy_probability)
        + (1.0 - entropy_probability) * np.log(1.0 - entropy_probability)
    )
    uncertainty = 1.0 - np.abs(2.0 * probability - 1.0)
    image_float = image.astype(np.float32)

    row: dict[str, object] = {"case_id": case_id}
    if group is not None:
        row["group"] = group
    row.update(
        {
            "gt_fg_ratio": float(gt.mean()),
            "gt_boundary_ratio": float(boundary.mean()),
            "gt_connected_components": int(ndimage.label(gt)[1]),
            "r_mean": float(image_float[..., 0].mean()),
            "g_mean": float(image_float[..., 1].mean()),
            "b_mean": float(image_float[..., 2].mean()),
            "r_std": float(image_float[..., 0].std()),
            "g_std": float(image_float[..., 1].std()),
            "b_std": float(image_float[..., 2].std()),
            "brightness_mean": float(image_float.mean(axis=2).mean()),
            "exg": float(
                np.mean(
                    2.0 * image_float[..., 1]
                    - image_float[..., 0]
                    - image_float[..., 2]
                )
            ),
            "soft_pred_fg_ratio": float(probability.mean()),
            "prob_std": float(probability.std()),
            "entropy_mean": float(entropy.mean()),
            "bg_uncertainty": float(uncertainty[background].mean()) if background.any() else 0.0,
            "hard_pred_fg_ratio": float(hard.mean()),
            "prediction_gap": float(hard.mean() - gt.mean()),
            "hard_fn_rate": float((~hard[gt]).mean()) if gt.any() else 0.0,
            "hard_fp_rate": float(hard[background].mean()) if background.any() else 0.0,
            "hard_boundary_error": (
                float(np.abs(hard[boundary].astype(np.float32) - gt[boundary].astype(np.float32)).mean())
                if boundary.any()
                else 0.0
            ),
            "pred_connected_components": int(ndimage.label(hard)[1]),
            "pred_boundary_ratio": float(hard_boundary.mean()),
            "image_path": str(image_path.resolve()),
            "label_path": str(label_path.resolve()),
            "probability_path": str(probability_path.resolve()),
            "prediction_path": str(hard_mask_path.resolve()),
        }
    )
    return row


def main() -> None:
    args = parse_args()
    if not 0.0 < args.threshold < 1.0:
        raise ValueError("threshold must be in (0, 1)")
    case_ids = load_case_ids(args.case_ids, args.probability_dir)
    images = index_by_stem(
        args.image_dir, IMAGE_EXTENSIONS, strip_nnunet_channel=True
    )
    labels = index_by_stem(args.label_dir, IMAGE_EXTENSIONS)
    probabilities = index_by_stem(args.probability_dir, (".npy",))
    hard_masks = index_by_stem(args.hard_mask_dir, IMAGE_EXTENSIONS)
    resolve_group = build_group_resolver(args)

    rows = []
    for index, case_id in enumerate(case_ids, start=1):
        missing = [
            name
            for name, table in (
                ("image", images),
                ("label", labels),
                ("probability", probabilities),
                ("hard_mask", hard_masks),
            )
            if case_id not in table
        ]
        if missing:
            raise FileNotFoundError(f"Missing {missing} for case {case_id!r}")
        rows.append(
            compute_row(
                case_id,
                images[case_id],
                labels[case_id],
                probabilities[case_id],
                hard_masks[case_id],
                args.threshold,
                resolve_group(case_id),
            )
        )
        if index % 100 == 0 or index == len(case_ids):
            print(f"features: {index}/{len(case_ids)}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(args.output), "case_count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
