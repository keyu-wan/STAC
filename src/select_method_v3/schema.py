"""Input schema constants for the E0 selector."""

LEGACY16_FEATURES = (
    "gt_fg_ratio",
    "gt_boundary_ratio",
    "hard_pred_fg_ratio",
    "pred_connected_components",
    "gt_connected_components",
    "soft_pred_fg_ratio",
    "prob_std",
    "entropy_mean",
    "r_mean",
    "g_mean",
    "b_mean",
    "r_std",
    "g_std",
    "b_std",
    "brightness_mean",
    "exg",
)

GT_MORPHOLOGY_FEATURES = (
    "gt_fg_ratio",
    "gt_boundary_ratio",
    "gt_connected_components",
)

WARMUP_PREDICTION_FEATURES = (
    "hard_pred_fg_ratio",
    "pred_connected_components",
    "soft_pred_fg_ratio",
    "prob_std",
    "entropy_mean",
)

APPEARANCE_FEATURES = (
    "r_mean",
    "g_mean",
    "b_mean",
    "r_std",
    "g_std",
    "b_std",
    "brightness_mean",
    "exg",
)

APPEARANCE_NO_EXG_FEATURES = tuple(
    feature for feature in APPEARANCE_FEATURES if feature != "exg"
)

LEGACY15_NO_EXG_FEATURES = tuple(
    feature for feature in LEGACY16_FEATURES if feature != "exg"
)

REPRESENTATION_MODES = (
    "legacy16",
    "legacy15_no_exg",
    "group_equal_with_exg",
    "group_equal_without_exg",
    "gt_appearance_equal_with_exg",
    "no_gt_morphology_view",
    "no_appearance_view",
)

STABLE_FEATURES = (
    "gt_fg_ratio",
    "gt_boundary_ratio",
    "gt_connected_components",
    "r_mean",
    "g_mean",
    "b_mean",
    "r_std",
    "g_std",
    "b_std",
    "brightness_mean",
    "exg",
)

FOREST_REQUIRED_FEATURES = (
    "hard_fp_rate",
    "bg_uncertainty",
    "hard_boundary_error",
    "prediction_gap",
)

REQUIRED_FEATURES = tuple(
    dict.fromkeys(("case_id",) + LEGACY16_FEATURES + FOREST_REQUIRED_FEATURES)
)

ARTIFACT_DEPENDENT_FEATURES = (
    "hard_pred_fg_ratio",
    "pred_connected_components",
    "soft_pred_fg_ratio",
    "prob_std",
    "entropy_mean",
    "hard_fp_rate",
    "bg_uncertainty",
    "hard_boundary_error",
    "prediction_gap",
)

METHOD_NAME = "v3_group_equal_with_exg_no_guard_no_bonus"
METHOD_VERSION = "3.2.0"

