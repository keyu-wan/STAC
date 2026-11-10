"""Portable CSV/JSON I/O with strict validation."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .schema import REQUIRED_FEATURES


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_feature_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = [name for name in REQUIRED_FEATURES if name not in columns]
        if missing:
            raise ValueError(f"Feature CSV is missing required columns: {missing}")
        rows: list[dict[str, Any]] = []
        for line_number, raw in enumerate(reader, start=2):
            case_id = (raw.get("case_id") or "").strip()
            if not case_id:
                raise ValueError(f"Empty case_id at CSV line {line_number}")
            row: dict[str, Any] = {"case_id": case_id}
            for key, value in raw.items():
                if key == "case_id":
                    continue
                if value is None or value == "":
                    row[key] = 0.0
                    continue
                try:
                    row[key] = float(value)
                except ValueError:
                    row[key] = value
            rows.append(row)
    rows.sort(key=lambda row: str(row["case_id"]))
    ids = [str(row["case_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Feature CSV contains duplicate case_id values")
    if not rows:
        raise ValueError("Feature CSV is empty")
    return rows


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    fields = sorted({key for row in materialized for key in row}) or ["case_id"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)

