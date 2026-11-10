#!/usr/bin/env python3
"""Convert a v3 selection JSON into a minimal nnU-Net-style split file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    args = parser.parse_args()
    payload = json.loads(args.selection.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != args.expected_count:
        raise ValueError(
            f"Expected {args.expected_count} selected cases, got "
            f"{len(cases) if isinstance(cases, list) else type(cases).__name__}"
        )
    if len(cases) != len(set(cases)):
        raise ValueError("Selection contains duplicate cases")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps([{"train": sorted(str(case) for case in cases), "val": []}], indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()

