"""Standalone E0 selection algorithm.

This module intentionally has no dependency on nnU-Net or project-specific
filesystem layouts.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .schema import (
    APPEARANCE_FEATURES,
    APPEARANCE_NO_EXG_FEATURES,
    GT_MORPHOLOGY_FEATURES,
    LEGACY15_NO_EXG_FEATURES,
    LEGACY16_FEATURES,
    METHOD_NAME,
    METHOD_VERSION,
    REPRESENTATION_MODES,
    STABLE_FEATURES,
    WARMUP_PREDICTION_FEATURES,
)


def finite(value: Any, default: float = 0.0) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return default
    return converted if np.isfinite(converted) else default


def normalize_column(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    scale = 1.4826 * mad
    if (not np.isfinite(scale)) or scale < 1e-6:
        std = float(values.std())
        scale = std if std > 1e-6 else 1.0
    return (values - median) / scale


def row_l2(matrix: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norm > 1e-8, norm, 1.0)


def matrix_from_columns(
    rows: Sequence[Mapping[str, Any]], features: Sequence[str]
) -> np.ndarray:
    columns = [
        normalize_column(
            np.asarray([finite(row.get(feature, 0.0)) for row in rows], dtype=np.float32)
        )
        for feature in features
    ]
    return row_l2(np.stack(columns, axis=1).astype(np.float32))


def legacy16_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return matrix_from_columns(rows, LEGACY16_FEATURES)


def group_equal_matrix(
    rows: Sequence[Mapping[str, Any]], *, include_exg: bool
) -> np.ndarray:
    """Build a representation whose three semantic views have equal norm."""

    appearance = APPEARANCE_FEATURES if include_exg else APPEARANCE_NO_EXG_FEATURES
    coefficient = np.float32(math.sqrt(1.0 / 3.0))
    groups = (
        matrix_from_columns(rows, GT_MORPHOLOGY_FEATURES),
        matrix_from_columns(rows, WARMUP_PREDICTION_FEATURES),
        matrix_from_columns(rows, appearance),
    )
    return np.concatenate([coefficient * group for group in groups], axis=1)


def gt_appearance_equal_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Build a representation using equal GT morphology and appearance views."""

    coefficient = np.float32(math.sqrt(1.0 / 2.0))
    groups = (
        matrix_from_columns(rows, GT_MORPHOLOGY_FEATURES),
        matrix_from_columns(rows, APPEARANCE_FEATURES),
    )
    return np.concatenate([coefficient * group for group in groups], axis=1)


def prediction_appearance_equal_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Ablation representation that removes the GT morphology view only."""

    coefficient = np.float32(math.sqrt(1.0 / 2.0))
    groups = (
        matrix_from_columns(rows, WARMUP_PREDICTION_FEATURES),
        matrix_from_columns(rows, APPEARANCE_FEATURES),
    )
    return np.concatenate([coefficient * group for group in groups], axis=1)


def gt_prediction_equal_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Ablation representation that removes the image appearance view only."""

    coefficient = np.float32(math.sqrt(1.0 / 2.0))
    groups = (
        matrix_from_columns(rows, GT_MORPHOLOGY_FEATURES),
        matrix_from_columns(rows, WARMUP_PREDICTION_FEATURES),
    )
    return np.concatenate([coefficient * group for group in groups], axis=1)


def representation_matrix(
    rows: Sequence[Mapping[str, Any]], mode: str
) -> np.ndarray:
    if mode == "legacy16":
        return legacy16_matrix(rows)
    if mode == "legacy15_no_exg":
        return matrix_from_columns(rows, LEGACY15_NO_EXG_FEATURES)
    if mode == "group_equal_with_exg":
        return group_equal_matrix(rows, include_exg=True)
    if mode == "group_equal_without_exg":
        return group_equal_matrix(rows, include_exg=False)
    if mode == "gt_appearance_equal_with_exg":
        return gt_appearance_equal_matrix(rows)
    if mode == "no_gt_morphology_view":
        return prediction_appearance_equal_matrix(rows)
    if mode == "no_appearance_view":
        return gt_prediction_equal_matrix(rows)
    raise ValueError(f"unknown representation_mode: {mode}")


def rank01(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if not len(array):
        return array
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.arange(len(array), dtype=np.float32)
    if len(array) == 1:
        return np.ones_like(ranks)
    return ranks / float(len(array) - 1)


def forest_indices(row: Mapping[str, Any]) -> dict[str, float]:
    red, green, blue = (
        finite(row.get(name, 0.0)) for name in ("r_mean", "g_mean", "b_mean")
    )
    return {
        "gli": float(
            np.clip(
                (2 * green - red - blue) / max(2 * green + red + blue, 1e-6),
                -1,
                1,
            )
        ),
        "green_cv": float(
            np.clip(finite(row.get("g_std", 0.0)) / max(green, 1e-6), 0, 4)
        ),
        "exg": finite(row.get("exg", 0.0)),
    }


def false_positive_risk(row: Mapping[str, Any]) -> float:
    foreground = finite(row.get("gt_fg_ratio"))
    fp_rate = finite(row.get("hard_fp_rate"))
    background_uncertainty = finite(row.get("bg_uncertainty"))
    predicted = finite(row.get("hard_pred_fg_ratio"))
    boundary = finite(row.get("hard_boundary_error"))
    gap = finite(row.get("prediction_gap"))
    if foreground <= 0.01:
        return (
            0.45 * fp_rate
            + 0.25 * background_uncertainty
            + 0.20 * predicted
            + 0.10 * boundary
        )
    return (
        0.35 * fp_rate
        + 0.20 * background_uncertainty
        + 0.20 * max(0.0, predicted - foreground)
        + 0.15 * boundary
        + 0.10 * gap
    )


def legacy_compact_forest_values(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, float], dict[str, float]]:
    foreground = rank01([finite(row.get("gt_fg_ratio")) for row in rows])
    boundary = rank01([finite(row.get("gt_boundary_ratio")) for row in rows])
    component_gap = rank01(
        [
            abs(
                finite(row.get("gt_connected_components"))
                - finite(row.get("pred_connected_components"))
            )
            for row in rows
        ]
    )
    indices = [forest_indices(row) for row in rows]
    gli = rank01([item["gli"] for item in indices])
    exg = rank01([item["exg"] for item in indices])
    green_cv = rank01([item["green_cv"] for item in indices])
    risks = np.asarray([false_positive_risk(row) for row in rows], np.float32)
    risk_rank = rank01(risks)

    forest: dict[str, float] = {}
    for index, row in enumerate(rows):
        foreground_presence = min(1.0, 2 * float(foreground[index]))
        green = (
            0.45 * float(gli[index])
            + 0.35 * float(exg[index])
            + 0.20 * float(green_cv[index])
        )
        risk = 0.5 * float(risk_rank[index]) + 0.5 * float(risks[index])
        value = (
            0.24 * foreground_presence
            + 0.24 * float(boundary[index])
            + 0.20 * float(component_gap[index])
            + 0.20 * green
            - 0.10 * risk
        )
        forest[str(row["case_id"])] = float(np.clip(value, 0, 1))

    stable_forest: dict[str, float] = {}
    gt_components = rank01(
        [finite(row.get("gt_connected_components")) for row in rows]
    )
    for index, row in enumerate(rows):
        green = (
            0.45 * float(gli[index])
            + 0.35 * float(exg[index])
            + 0.20 * float(green_cv[index])
        )
        stable_forest[str(row["case_id"])] = float(
            np.clip(
                0.35 * foreground[index]
                + 0.35 * boundary[index]
                + 0.20 * gt_components[index]
                + 0.10 * green,
                0,
                1,
            )
        )
    return forest, stable_forest


def similarity(matrix: np.ndarray, representatives: Sequence[int]) -> np.ndarray:
    if not representatives:
        return np.zeros((len(matrix), 0), np.float32)
    selected = matrix[np.asarray(representatives)]
    return np.clip(0.5 * (matrix @ selected.T + 1), 0, 1).astype(np.float32)


def local_similarity(matrix: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    local = matrix[np.asarray(indices)]
    return np.clip(0.5 * (local @ local.T + 1), 0, 1).astype(np.float32)


def normalized_forest(
    case_ids: Sequence[str], forest: Mapping[str, float]
) -> dict[str, float]:
    values = np.asarray([forest[case_id] for case_id in case_ids], np.float32)
    low, high = np.quantile(values, [0.1, 0.9])
    denominator = max(float(high - low), 1e-6)
    return {
        case_id: float(np.clip((forest[case_id] - low) / denominator, 0, 1))
        for case_id in case_ids
    }


def infer_group(row: Mapping[str, Any]) -> str:
    explicit = str(row.get("group", "")).strip()
    if explicit and explicit != "0.0":
        return explicit
    case_id = str(row["case_id"])
    match = re.search(r"(?:^|_)(rural|urban)(?:_|$)", case_id, re.IGNORECASE)
    return match.group(1).title() if match else "All"


def adaptive_positive_bin_edges(
    values: Sequence[float], max_bins: int = 5
) -> list[float]:
    positive = np.asarray([value for value in values if value > 0], dtype=float)
    if not len(positive):
        return []
    spread = float(np.std(np.log(positive)))
    count_driven = int(np.clip(round(math.sqrt(len(positive)) / 12), 2, max_bins))
    spread_driven = (
        2
        if spread < 0.75
        else 3
        if spread < 1.25
        else 4
        if spread < 1.75
        else max_bins
    )
    bin_count = int(np.clip(max(count_driven, spread_driven), 2, max_bins))
    quantiles = np.linspace(0.0, 1.0, bin_count + 1)[1:-1]
    return [
        float(value)
        for value in np.unique(np.exp(np.quantile(np.log(positive), quantiles)))
    ]


def distribution_tau(counts: Mapping[str, int]) -> float:
    values = np.asarray([value for value in counts.values() if value > 0], dtype=float)
    if len(values) <= 1:
        return 1.0
    probabilities = values / values.sum()
    entropy = float(
        -(probabilities * np.log(np.clip(probabilities, 1e-8, 1.0))).sum()
    )
    return float(np.clip(entropy / np.log(len(values)), 0.6, 1.0))


def allocate_proportional_quotas(
    counts: Mapping[str, int],
    budget: int,
    background_ratio: float,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Allocate quotas by empirical stratum frequency with the same bg cap."""
    active = {str(key): int(value) for key, value in counts.items() if value > 0}
    if not active:
        return {}, {"background_cap": 0}
    if budget < 1 or budget > sum(active.values()):
        raise ValueError("budget must be between 1 and the number of cases")
    total = sum(active.values())
    raw = {key: budget * active[key] / total for key in active}
    quotas = {key: min(int(math.floor(raw[key])), active[key]) for key in active}
    if budget >= len(active):
        quotas = {key: max(1, value) for key, value in quotas.items()}

    background = sorted(key for key in active if key.endswith("::bg_only"))
    cap = int(math.floor(budget * background_ratio + 1e-9))
    while background and sum(quotas[key] for key in background) > cap:
        key = max(background, key=lambda item: (quotas[item], item))
        quotas[key] -= 1

    fractions = {key: raw[key] - math.floor(raw[key]) for key in active}
    while sum(quotas.values()) < budget:
        background_used = sum(quotas[key] for key in background)
        candidates = [
            key
            for key in active
            if quotas[key] < active[key]
            and (key not in background or background_used < cap)
        ]
        if not candidates:
            raise RuntimeError("No stratum can receive the remaining quota")
        winner = max(
            candidates,
            key=lambda key: (fractions[key], active[key] - quotas[key], key),
        )
        quotas[winner] += 1
    return quotas, {
        "background_cap": cap,
        "background_ratio": background_ratio,
        "raw_quotas": raw,
    }


def allocate_gt_metadata_quotas(
    counts: Mapping[str, int],
    budget: int,
    background_ratio: float,
    stratum_value: Mapping[str, float],
    value_weight: float,
) -> tuple[dict[str, int], dict[str, Any]]:
    """Allocate quotas using only GT/image metadata and legacy residual fill."""
    active = {str(key): int(value) for key, value in counts.items() if value > 0}
    if not active:
        return {}, {"tau": 1.0, "background_cap": 0}
    if budget < 1 or budget > sum(active.values()):
        raise ValueError("budget must be between 1 and the number of cases")
    tau = distribution_tau(active)
    weights = {key: value**tau for key, value in active.items()}
    total = sum(weights.values())
    raw = {key: budget * weights[key] / total for key in active}
    quotas = {key: min(int(math.floor(raw[key])), active[key]) for key in active}
    if budget >= len(active):
        quotas = {key: max(1, value) for key, value in quotas.items()}

    background = sorted(key for key in active if key.endswith("::bg_only"))
    cap = int(math.floor(budget * background_ratio + 1e-9))
    while background and sum(quotas[key] for key in background) > cap:
        key = max(background, key=lambda item: (quotas[item], item))
        quotas[key] -= 1

    values = np.asarray(
        [finite(stratum_value.get(key)) for key in active], dtype=float
    )
    low, high = float(values.min()), float(values.max())
    normalized_value = {
        key: (finite(stratum_value.get(key)) - low) / max(high - low, 1e-12)
        for key in active
    }
    fractions = {key: raw[key] - math.floor(raw[key]) for key in active}
    while sum(quotas.values()) < budget:
        background_used = sum(quotas[key] for key in background)
        candidates = [
            key
            for key in active
            if quotas[key] < active[key]
            and (key not in background or background_used < cap)
        ]
        if not candidates:
            raise RuntimeError("No stratum can receive the remaining quota")
        winner = max(
            candidates,
            key=lambda key: (
                fractions[key] + value_weight * normalized_value[key],
                active[key] - quotas[key],
                key,
            ),
        )
        quotas[winner] += 1
    return quotas, {
        "tau": tau,
        "background_cap": cap,
        "background_ratio": background_ratio,
        "raw_quotas": raw,
    }


@dataclass(frozen=True)
class E0Config:
    retention: float = 0.10
    forest_bonus_weight: float = 0.0
    forest_bonus_cap: float = 0.0
    forest_bonus_mode: str = "absolute_capped"
    guard_swap_ratio: float = 0.0
    guard_coverage_tolerance: float = 0.02
    gt_background_scale: float = 0.685
    gt_value_weight: float = 0.10
    max_foreground_bins: int = 5
    representation_mode: str = "group_equal_with_exg"
    quota_mode: str = "gt_metadata_adaptive_legacyfill"
    selection_mode: str = "facility_location"
    random_seed: int = 0

    def validate(self) -> None:
        if not 0 < self.retention <= 1:
            raise ValueError("retention must be in (0, 1]")
        if self.forest_bonus_weight < 0 or self.forest_bonus_cap < 0:
            raise ValueError("forest bonus parameters must be non-negative")
        if self.forest_bonus_mode not in {
            "absolute_capped",
            "coverage_scaled_linear",
        }:
            raise ValueError("unknown forest_bonus_mode")
        if not 0 <= self.guard_swap_ratio <= 1:
            raise ValueError("guard_swap_ratio must be in [0, 1]")
        if self.guard_coverage_tolerance < 0:
            raise ValueError("guard_coverage_tolerance must be non-negative")
        if self.gt_background_scale < 0:
            raise ValueError("gt_background_scale must be non-negative")
        if self.gt_value_weight < 0:
            raise ValueError("gt_value_weight must be non-negative")
        if self.max_foreground_bins < 2:
            raise ValueError("max_foreground_bins must be at least 2")
        if self.representation_mode not in REPRESENTATION_MODES:
            raise ValueError("unknown representation_mode")
        if self.quota_mode not in {"gt_metadata_adaptive_legacyfill", "proportional"}:
            raise ValueError("unknown quota_mode")
        if self.selection_mode not in {"facility_location", "random_within_strata"}:
            raise ValueError("unknown selection_mode")


@dataclass
class SelectionContext:
    rows: list[dict[str, Any]]
    case_ids: list[str]
    case_to_index: dict[str, int]
    primary_matrix: np.ndarray
    stable_matrix: np.ndarray
    forest: dict[str, float]
    stable_forest: dict[str, float]
    strata: dict[str, list[dict[str, Any]]]
    quotas: dict[str, int]
    quota_metadata: dict[str, Any]


@dataclass
class SelectionResult:
    pre_guard_cases: list[str]
    cases: list[str]
    trace: list[dict[str, Any]]
    summary: dict[str, Any]


def build_context(
    input_rows: Sequence[Mapping[str, Any]],
    config: E0Config,
) -> SelectionContext:
    input_by_case = {str(row["case_id"]): dict(row) for row in input_rows}
    if len(input_by_case) != len(input_rows):
        raise ValueError("Rows contain duplicate case IDs")

    case_ids = sorted(input_by_case)
    rows = [input_by_case[case_id] for case_id in case_ids]
    forest, stable_forest = legacy_compact_forest_values(rows)
    edges = adaptive_positive_bin_edges(
        [finite(row.get("gt_fg_ratio")) for row in rows],
        config.max_foreground_bins,
    )
    strata: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        group = infer_group(row)
        foreground = finite(row.get("gt_fg_ratio"))
        if foreground <= 0:
            fg_bin = "bg_only"
        else:
            fg_bin = (
                f"fg_q{int(np.searchsorted(edges, foreground, side='right')) + 1}"
            )
        row.update(
            {
                "group": group,
                "fg_bin": fg_bin,
                "stratum": f"{group.lower()}::{fg_bin}",
            }
        )
        strata.setdefault(str(row["stratum"]), []).append(row)
    counts = {name: len(items) for name, items in strata.items()}
    budget = max(1, min(len(rows), int(round(len(rows) * config.retention))))
    background_count = sum(
        count for name, count in counts.items() if name.endswith("::bg_only")
    )
    full_background_ratio = background_count / len(rows)
    stratum_value = {
        name: float(
            sum(stable_forest[str(row["case_id"])] for row in items)
        )
        for name, items in strata.items()
    }
    quota_background_ratio = full_background_ratio * config.gt_background_scale
    if config.quota_mode == "proportional":
        quotas, allocation = allocate_proportional_quotas(
            counts,
            budget,
            quota_background_ratio,
        )
    else:
        quotas, allocation = allocate_gt_metadata_quotas(
            counts,
            budget,
            quota_background_ratio,
            stratum_value,
            config.gt_value_weight,
        )
    primary = representation_matrix(rows, config.representation_mode)
    stable = matrix_from_columns(rows, STABLE_FEATURES)
    unavailable = {
        name: target - len(strata.get(name, []))
        for name, target in quotas.items()
        if target > len(strata.get(name, []))
    }
    if unavailable:
        raise ValueError(f"Quotas exceed available cases: {unavailable}")

    return SelectionContext(
        rows=rows,
        case_ids=case_ids,
        case_to_index={case_id: index for index, case_id in enumerate(case_ids)},
        primary_matrix=primary,
        stable_matrix=stable,
        forest=forest,
        stable_forest=stable_forest,
        strata=strata,
        quotas=quotas,
        quota_metadata={
            "profile": config.quota_mode,
            "foreground_bin_edges": edges,
            "stratum_counts": counts,
            "full_background_ratio": full_background_ratio,
            "gt_background_scale": config.gt_background_scale,
            "gt_value_weight": config.gt_value_weight,
            **allocation,
        },
    )


def stable_random_score(seed: int, stratum: str, case_id: str) -> int:
    payload = f"{seed}|{stratum}|{case_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def select_random_within_strata(
    context: SelectionContext, config: E0Config
) -> tuple[list[str], list[dict[str, Any]]]:
    if config.selection_mode == "random_within_strata":
        return select_random_within_strata(context, config)

    selected_all: list[str] = []
    trace: list[dict[str, Any]] = []
    for stratum in sorted(context.quotas):
        target = context.quotas[stratum]
        rows = context.strata.get(stratum, [])
        if target <= 0:
            continue
        ranked = sorted(
            [str(row["case_id"]) for row in rows],
            key=lambda case_id: (stable_random_score(config.random_seed, stratum, case_id), case_id),
        )
        for step, case_id in enumerate(ranked[:target]):
            selected_all.append(case_id)
            trace.append(
                {
                    "case_id": case_id,
                    "stratum": stratum,
                    "group": rows[0]["group"],
                    "fg_bin": rows[0]["fg_bin"],
                    "stage": "random_within_strata",
                    "step": step,
                    "objective": 0.0,
                    "coverage_gain": 0.0,
                    "stable_coverage_gain": 0.0,
                    "forest_bonus": 0.0,
                    "forest_value": context.forest[case_id],
                    "forest_value_normalized": 0.0,
                    "quality": 0.0,
                    "tie_break_reason": f"sha256_seed_{config.random_seed}",
                }
            )
    return selected_all, trace


def select_core(
    context: SelectionContext, config: E0Config
) -> tuple[list[str], list[dict[str, Any]]]:
    selected_all: list[str] = []
    trace: list[dict[str, Any]] = []
    for stratum in sorted(context.quotas):
        target = context.quotas[stratum]
        case_ids = [
            str(row["case_id"]) for row in context.strata.get(stratum, [])
        ]
        if target <= 0:
            continue
        indices = [context.case_to_index[case_id] for case_id in case_ids]
        pairwise = local_similarity(context.primary_matrix, indices)
        stable_pairwise = local_similarity(context.stable_matrix, indices)
        best = np.zeros(len(case_ids), np.float32)
        best_stable = np.zeros(len(case_ids), np.float32)
        chosen: set[int] = set()
        forest_norm = normalized_forest(case_ids, context.forest)

        for step in range(target):
            candidates: list[dict[str, Any]] = []
            for position, case_id in enumerate(case_ids):
                if position in chosen:
                    continue
                new = np.maximum(best, pairwise[:, position])
                new_stable = np.maximum(best_stable, stable_pairwise[:, position])
                gain = float((new - best).sum() / len(case_ids))
                stable_gain = float(
                    (new_stable - best_stable).sum() / len(case_ids)
                )
                if config.forest_bonus_mode == "absolute_capped":
                    bonus = min(
                        config.forest_bonus_cap,
                        config.forest_bonus_weight * forest_norm[case_id],
                    )
                else:
                    bonus = (
                        gain
                        * config.forest_bonus_weight
                        * forest_norm[case_id]
                    )
                candidates.append(
                    {
                        "pos": position,
                        "case_id": case_id,
                        "objective": gain + bonus,
                        "coverage_gain": gain,
                        "stable_coverage_gain": stable_gain,
                        "forest_bonus": bonus,
                        "forest_value": context.forest[case_id],
                        "forest_value_normalized": forest_norm[case_id],
                        "quality": float(new.mean()),
                    }
                )
            ranked = sorted(
                candidates, key=lambda candidate: (-candidate["objective"], candidate["pos"])
            )
            # E0 has no non-zero robustness radius, but its legacy candidate
            # core still applies the stable fallback when objective values are
            # exactly equal. Keeping this branch is necessary for bit-exact
            # replay and gives exact ties an artifact-independent resolution.
            tied = [
                candidate
                for candidate in candidates
                if ranked[0]["objective"] - candidate["objective"] <= 0.0
            ]
            if len(tied) > 1:
                tied.sort(
                    key=lambda candidate: (
                        -candidate["stable_coverage_gain"],
                        -context.stable_forest[candidate["case_id"]],
                        candidate["case_id"],
                    )
                )
                winner = tied[0]
                tie_break_reason = (
                    "exact_objective_tie>stable_gain>stable_forest>case_id"
                )
            else:
                winner = ranked[0]
                tie_break_reason = "objective"
            chosen.add(int(winner["pos"]))
            best = np.maximum(best, pairwise[:, int(winner["pos"])])
            best_stable = np.maximum(
                best_stable, stable_pairwise[:, int(winner["pos"])]
            )
            second = (
                ranked[1]["objective"] if len(ranked) > 1 else ranked[0]["objective"]
            )
            trace.append(
                {
                    **winner,
                    "stratum": stratum,
                    "group": context.strata[stratum][0]["group"],
                    "fg_bin": context.strata[stratum][0]["fg_bin"],
                    "stage": "core",
                    "step": step,
                    "exact_tie_candidate_count": len(tied),
                    "selection_margin": ranked[0]["objective"] - second,
                    "tie_break_reason": tie_break_reason,
                }
            )
            selected_all.append(str(winner["case_id"]))
    return selected_all, trace


@dataclass
class CoverageCache:
    selected_ids: list[str]
    top1: np.ndarray
    top2: np.ndarray
    top1_owner: np.ndarray

    @classmethod
    def build(
        cls,
        selected_ids: Sequence[str],
        context: SelectionContext,
    ) -> "CoverageCache":
        selected = list(selected_ids)
        representatives = [context.case_to_index[case_id] for case_id in selected]
        similarities = similarity(context.primary_matrix, representatives)
        if similarities.shape[1] == 0:
            zeros = np.zeros(len(context.rows), np.float32)
            return cls(
                selected,
                zeros,
                zeros.copy(),
                np.full(len(zeros), -1, np.int32),
            )
        order = np.argsort(similarities, axis=1, kind="stable")
        row_indices = np.arange(len(similarities))
        owner = order[:, -1]
        top1 = similarities[row_indices, owner]
        top2 = (
            similarities[row_indices, order[:, -2]]
            if similarities.shape[1] > 1
            else np.zeros_like(top1)
        )
        return cls(selected, top1, top2, owner)

    @property
    def quality(self) -> float:
        return float(self.top1.mean()) if self.top1.size else 0.0

    def similarity_to(
        self, case_id: str, context: SelectionContext
    ) -> np.ndarray:
        return similarity(
            context.primary_matrix, [context.case_to_index[case_id]]
        )[:, 0]

    def trial_quality(self, outgoing: str, incoming_similarity: np.ndarray) -> float:
        position = self.selected_ids.index(outgoing)
        without = np.where(self.top1_owner == position, self.top2, self.top1)
        return float(np.maximum(without, incoming_similarity).mean())


def legacy_shift_stats(
    context: SelectionContext,
) -> dict[str, tuple[float, float]]:
    fields = (
        "gt_fg_ratio",
        "gt_boundary_ratio",
        "hard_pred_fg_ratio",
        "gt_connected_components",
        "exg",
    )
    return {
        feature: (
            float(np.mean([finite(row.get(feature)) for row in context.rows])),
            max(
                float(np.std([finite(row.get(feature)) for row in context.rows])),
                1e-6,
            ),
        )
        for feature in fields
    }


def legacy_shift(
    case_ids: Sequence[str],
    context: SelectionContext,
    stats: Mapping[str, tuple[float, float]],
) -> float:
    by_case = {str(row["case_id"]): row for row in context.rows}
    total = 0.0
    for feature, (mean, std) in stats.items():
        selected_mean = float(
            np.mean([finite(by_case[case_id].get(feature)) for case_id in case_ids])
        )
        total += abs(selected_mean - mean) / std
    return total / len(stats)


def legacy_guard(
    context: SelectionContext,
    selected_cases: Sequence[str],
    max_swaps: int,
    coverage_tolerance: float,
) -> tuple[list[str], list[dict[str, Any]], dict[str, Any]]:
    selected = list(selected_cases)
    chosen = set(selected)
    by_case = {str(row["case_id"]): row for row in context.rows}
    stats = legacy_shift_stats(context)
    shift = legacy_shift(selected, context, stats)
    cache = CoverageCache.build(selected, context)
    coverage = cache.quality
    initial_shift = shift
    initial_coverage = coverage
    trace: list[dict[str, Any]] = []

    for step in range(max_swaps):
        best: tuple[Any, ...] | None = None
        incoming_cache: dict[str, np.ndarray] = {}
        for stratum, rows in context.strata.items():
            outgoing_ids = [
                case_id
                for case_id in selected
                if by_case[case_id]["stratum"] == stratum
            ]
            incoming_ids = [
                str(row["case_id"])
                for row in rows
                if str(row["case_id"]) not in chosen
            ]

            def distance(case_id: str) -> float:
                row = by_case[case_id]
                return abs(
                    finite(row["gt_fg_ratio"]) - stats["gt_fg_ratio"][0]
                ) + abs(finite(row["exg"]) - stats["exg"][0])

            for outgoing in sorted(
                outgoing_ids, key=distance, reverse=True
            )[:4]:
                for incoming in sorted(incoming_ids, key=distance)[:6]:
                    trial = [
                        case_id for case_id in selected if case_id != outgoing
                    ] + [incoming]
                    new_shift = legacy_shift(trial, context, stats)
                    if incoming not in incoming_cache:
                        incoming_cache[incoming] = cache.similarity_to(
                            incoming, context
                        )
                    new_coverage = cache.trial_quality(
                        outgoing, incoming_cache[incoming]
                    )
                    if (
                        new_coverage < coverage - coverage_tolerance
                        or new_shift >= shift - 1e-8
                    ):
                        continue
                    key = (new_shift, -new_coverage, outgoing, incoming)
                    if best is None or key < best[0]:
                        best = (
                            key,
                            outgoing,
                            incoming,
                            new_shift,
                            new_coverage,
                            stratum,
                        )
        if best is None:
            break
        _, outgoing, incoming, shift, coverage, stratum = best
        selected = [
            case_id for case_id in selected if case_id != outgoing
        ] + [incoming]
        chosen.remove(outgoing)
        chosen.add(incoming)
        trace.extend(
            [
                {
                    "case_id": outgoing,
                    "paired_case_id": incoming,
                    "stratum": stratum,
                    "stage": "guard_swap_out",
                    "step": step,
                },
                {
                    "case_id": incoming,
                    "paired_case_id": outgoing,
                    "stratum": stratum,
                    "stage": "guard_swap_in",
                    "step": step,
                },
            ]
        )
        cache = CoverageCache.build(selected, context)

    return (
        sorted(selected),
        trace,
        {
            "guard_mode": "legacy",
            "guard_swap_count": len(trace) // 2,
            "initial_shift_score": initial_shift,
            "final_shift_score": shift,
            "initial_coverage_quality": initial_coverage,
            "final_coverage_quality": coverage,
        },
    )


def coverage(
    case_ids: Sequence[str],
    context: SelectionContext,
    stable: bool = False,
) -> float:
    matrix = context.stable_matrix if stable else context.primary_matrix
    representatives = [context.case_to_index[case_id] for case_id in case_ids]
    similarities = similarity(matrix, representatives)
    return float(similarities.max(axis=1).mean()) if similarities.shape[1] else 0.0


def select(
    rows: Sequence[Mapping[str, Any]],
    config: E0Config | None = None,
) -> SelectionResult:
    config = config or E0Config()
    config.validate()
    context = build_context(rows, config)
    pre_guard, core_trace = select_core(context, config)
    budget = sum(context.quotas.values())
    if len(pre_guard) != budget or len(set(pre_guard)) != budget:
        raise RuntimeError(
            f"Core selector returned {len(pre_guard)} cases for budget {budget}"
        )

    max_swaps = int(round(budget * config.guard_swap_ratio))
    if max_swaps > 0:
        final, guard_trace, guard_summary = legacy_guard(
            context,
            pre_guard,
            max_swaps,
            config.guard_coverage_tolerance,
        )
    else:
        final = sorted(pre_guard)
        guard_trace = []
        guard_summary = {"guard_mode": "none", "guard_swap_count": 0}

    if len(final) != budget or len(set(final)) != budget:
        raise RuntimeError(
            f"Final selector returned {len(final)} cases for budget {budget}"
        )

    summary = {
        "method": METHOD_NAME,
        "method_version": METHOD_VERSION,
        "normalization": "artifact_fit",
        "representation": config.representation_mode,
        "selection_mode": config.selection_mode,
        "forest_value": "legacy_compact",
        "forest_bonus": (
            "none"
            if config.forest_bonus_weight == 0 or config.forest_bonus_cap == 0
            else (
                "min(cap, weight * normalized_forest)"
                if config.forest_bonus_mode == "absolute_capped"
                else "coverage_gain * weight * normalized_forest"
            )
        ),
        "artifact_runtime": "single",
        "retention": config.retention,
        "budget": budget,
        "config": asdict(config),
        "quotas": context.quotas,
        "quota_generation": context.quota_metadata,
        "pre_guard_primary_coverage": coverage(pre_guard, context),
        "post_guard_primary_coverage": coverage(final, context),
        "pre_guard_stable_coverage": coverage(pre_guard, context, stable=True),
        "post_guard_stable_coverage": coverage(final, context, stable=True),
        "pre_guard_mean_forest_value": float(
            np.mean([context.forest[case_id] for case_id in pre_guard])
        ),
        "post_guard_mean_forest_value": float(
            np.mean([context.forest[case_id] for case_id in final])
        ),
        **guard_summary,
    }
    return SelectionResult(
        pre_guard_cases=sorted(pre_guard),
        cases=sorted(final),
        trace=core_trace + guard_trace,
        summary=summary,
    )
