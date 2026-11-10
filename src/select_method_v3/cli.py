"""Command-line interface for select-method-v3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import E0Config, select
from .io import (
    load_feature_csv,
    save_json,
    sha256_file,
    write_csv,
)
from .schema import REPRESENTATION_MODES


OUTPUT_FILES = (
    "pre_guard_selection.json",
    "selection.json",
    "trace.csv",
    "summary.json",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="select-method-v3",
        description="Run the final v3 group-equal-with-ExG subset selector.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    command = subparsers.add_parser("select", help="select a subset")
    command.add_argument("--features", type=Path, required=True)
    command.add_argument("--retention", type=float, default=0.10)
    command.add_argument("--forest-bonus-weight", type=float, default=0.0)
    command.add_argument("--forest-bonus-cap", type=float, default=0.0)
    command.add_argument(
        "--forest-bonus-mode",
        choices=("absolute_capped", "coverage_scaled_linear"),
        default="absolute_capped",
    )
    command.add_argument("--guard-swap-ratio", type=float, default=0.0)
    command.add_argument("--guard-coverage-tolerance", type=float, default=0.02)
    command.add_argument("--gt-background-scale", type=float, default=0.685)
    command.add_argument("--gt-value-weight", type=float, default=0.10)
    command.add_argument("--max-foreground-bins", type=int, default=5)
    command.add_argument(
        "--representation-mode",
        choices=REPRESENTATION_MODES,
        default="group_equal_with_exg",
    )
    command.add_argument(
        "--quota-mode",
        choices=("gt_metadata_adaptive_legacyfill", "proportional"),
        default="gt_metadata_adaptive_legacyfill",
    )
    command.add_argument(
        "--selection-mode",
        choices=("facility_location", "random_within_strata"),
        default="facility_location",
    )
    command.add_argument("--random-seed", type=int, default=0)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument(
        "--overwrite",
        action="store_true",
        help="replace known v3 output files if they already exist",
    )

    compare = subparsers.add_parser(
        "compare", help="compare two selection JSON files"
    )
    compare.add_argument("--left", type=Path, required=True)
    compare.add_argument("--right", type=Path, required=True)
    compare.add_argument("--output", type=Path)
    return parser


def _run_select(args: argparse.Namespace) -> int:
    if not args.features.is_file():
        raise FileNotFoundError(args.features)
    existing = [
        args.output_dir / filename
        for filename in OUTPUT_FILES
        if (args.output_dir / filename).exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output files already exist; pass --overwrite to replace them: "
            + ", ".join(str(path) for path in existing)
        )

    rows = load_feature_csv(args.features)
    config = E0Config(
        retention=args.retention,
        forest_bonus_weight=args.forest_bonus_weight,
        forest_bonus_cap=args.forest_bonus_cap,
        forest_bonus_mode=args.forest_bonus_mode,
        guard_swap_ratio=args.guard_swap_ratio,
        guard_coverage_tolerance=args.guard_coverage_tolerance,
        gt_background_scale=args.gt_background_scale,
        gt_value_weight=args.gt_value_weight,
        max_foreground_bins=args.max_foreground_bins,
        representation_mode=args.representation_mode,
        quota_mode=args.quota_mode,
        selection_mode=args.selection_mode,
        random_seed=args.random_seed,
    )
    result = select(rows, config)
    summary = {
        **result.summary,
        "inputs": {
            "features": str(args.features.resolve()),
            "features_sha256": sha256_file(args.features),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        args.output_dir / "pre_guard_selection.json",
        {"count": len(result.pre_guard_cases), "cases": result.pre_guard_cases},
    )
    save_json(
        args.output_dir / "selection.json",
        {"count": len(result.cases), "cases": result.cases},
    )
    write_csv(args.output_dir / "trace.csv", result.trace)
    save_json(args.output_dir / "summary.json", summary)
    print(
        json.dumps(
            {
                "selection": str(args.output_dir / "selection.json"),
                "pre_guard_selection": str(
                    args.output_dir / "pre_guard_selection.json"
                ),
                "summary": str(args.output_dir / "summary.json"),
            },
            indent=2,
        )
    )
    return 0


def _load_cases(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    cases = payload.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"{path} must contain a cases list")
    if len(cases) != len(set(cases)):
        raise ValueError(f"{path} contains duplicate cases")
    return {str(case) for case in cases}


def _run_compare(args: argparse.Namespace) -> int:
    left = _load_cases(args.left)
    right = _load_cases(args.right)
    union = left | right
    payload = {
        "left_count": len(left),
        "right_count": len(right),
        "intersection": len(left & right),
        "union": len(union),
        "jaccard": len(left & right) / len(union) if union else 1.0,
        "same_set": left == right,
        "only_left": sorted(left - right),
        "only_right": sorted(right - left),
    }
    if args.output:
        save_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "select":
        return _run_select(args)
    if args.command == "compare":
        return _run_compare(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())

