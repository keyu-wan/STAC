#!/usr/bin/env python3
"""Convert paired RGB images and semantic masks to an nnU-Net 2D dataset."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--train-ids", type=Path, required=True)
    parser.add_argument("--val-ids", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--foreground-value", type=int, required=True)
    parser.add_argument("--test-images", type=Path)
    parser.add_argument("--test-labels", type=Path)
    return parser.parse_args()


def read_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    ids = [case_id for case_id in ids if case_id]
    if not ids:
        raise ValueError(f"No case IDs in {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate case IDs in {path}")
    return ids


def index_files(directory: Path, *, images: bool) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    result: dict[str, Path] = {}
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        case_id = path.stem
        if images and case_id.endswith("_0000"):
            case_id = case_id[:-5]
        if case_id in result:
            raise ValueError(f"Duplicate case ID {case_id!r} in {directory}")
        result[case_id] = path
    return result


def save_case(
    case_id: str,
    image_path: Path,
    label_path: Path,
    image_output: Path,
    label_output: Path,
    foreground_value: int,
) -> None:
    with Image.open(image_path) as source:
        source.convert("RGB").save(image_output, format="PNG")
    with Image.open(label_path) as source:
        labels = np.asarray(source)
    if labels.ndim != 2:
        raise ValueError(f"Expected a single-channel label mask for {case_id}")
    binary = (labels == foreground_value).astype(np.uint8)
    Image.fromarray(binary).save(label_output, format="PNG")


def copy_cases(
    case_ids: list[str],
    images: dict[str, Path],
    labels: dict[str, Path],
    image_dir: Path,
    label_dir: Path,
    foreground_value: int,
) -> None:
    missing = [
        case_id
        for case_id in case_ids
        if case_id not in images or case_id not in labels
    ]
    if missing:
        raise FileNotFoundError(f"Missing image/label pairs for: {missing[:10]}")
    for index, case_id in enumerate(case_ids, start=1):
        save_case(
            case_id,
            images[case_id],
            labels[case_id],
            image_dir / f"{case_id}_0000.png",
            label_dir / f"{case_id}.png",
            foreground_value,
        )
        if index % 100 == 0 or index == len(case_ids):
            print(f"prepared: {index}/{len(case_ids)}", flush=True)


def main() -> None:
    args = parse_args()
    if not re.fullmatch(r"Dataset\d{3,}_[-A-Za-z0-9]+", args.dataset_name):
        raise ValueError(
            "--dataset-name must look like Dataset501_STAC and use an unused ID"
        )
    if (args.test_images is None) != (args.test_labels is None):
        raise ValueError("--test-images and --test-labels must be provided together")

    train_ids = read_ids(args.train_ids)
    val_ids = read_ids(args.val_ids)
    if set(train_ids) & set(val_ids):
        raise ValueError("Training and validation IDs overlap")
    all_trainval = train_ids + val_ids
    output = args.output_root / args.dataset_name
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {output}")

    image_index = index_files(args.images, images=True)
    label_index = index_files(args.labels, images=False)
    output.mkdir(parents=True)
    for folder in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (output / folder).mkdir()
    copy_cases(
        all_trainval,
        image_index,
        label_index,
        output / "imagesTr",
        output / "labelsTr",
        args.foreground_value,
    )

    test_count = 0
    if args.test_images is not None and args.test_labels is not None:
        test_image_index = index_files(args.test_images, images=True)
        test_label_index = index_files(args.test_labels, images=False)
        test_ids = sorted(test_image_index)
        if set(test_ids) & set(all_trainval):
            raise ValueError("Test IDs overlap with train/validation IDs")
        copy_cases(
            test_ids,
            test_image_index,
            test_label_index,
            output / "imagesTs",
            output / "labelsTs",
            args.foreground_value,
        )
        test_count = len(test_ids)

    dataset_json = {
        "channel_names": {"0": "R", "1": "G", "2": "B"},
        "labels": {"background": 0, "forest": 1},
        "numTraining": len(all_trainval),
        "file_ending": ".png",
        "name": args.dataset_name,
        "overwrite_image_reader_writer": "NaturalImage2DIO",
    }
    (output / "dataset.json").write_text(
        json.dumps(dataset_json, indent=2) + "\n", encoding="utf-8"
    )
    (output / "splits_final.json").write_text(
        json.dumps([{"train": train_ids, "val": val_ids}], indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "dataset": str(output),
                "train_cases": len(train_ids),
                "validation_cases": len(val_ids),
                "test_cases": test_count,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
