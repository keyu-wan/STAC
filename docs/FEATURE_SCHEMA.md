# Feature CSV schema

The selector expects one CSV row per candidate training case.

## Columns used by the paper's final configuration

The final `group_equal_with_exg` representation uses three equal-weight views:
ground-truth morphology, warm-up prediction behavior, and image appearance.
The CSV requires `case_id` and the following 16 numeric features:

| Column | Meaning |
|---|---|
| gt_fg_ratio | ground-truth foreground fraction |
| gt_boundary_ratio | ground-truth boundary-band fraction |
| gt_connected_components | ground-truth component count |
| hard_pred_fg_ratio | thresholded predicted foreground fraction |
| pred_connected_components | predicted component count |
| soft_pred_fg_ratio | mean foreground probability |
| prob_std | probability standard deviation |
| entropy_mean | mean predictive entropy |
| r_mean, g_mean, b_mean | channel means |
| r_std, g_std, b_std | channel standard deviations |
| brightness_mean | image brightness mean |
| exg | excess-green statistic |

These same GT and appearance features also drive the stable tie-break and
quota-value calculation.

An optional string `group` column may define the metadata domain. If absent,
LoveDA `Rural`/`Urban` is inferred from `case_id`; otherwise the generic group
`All` is used. Foreground bins are log-quantile bins computed from
`gt_fg_ratio`. Stratum quotas are generated from counts, the GT-only background
rate, and stable GT/image forest values. No external reference file is used.

The generic builder `scripts/build_features.py` can write this optional `group`
column from command-line metadata sources: a metadata CSV column, a constant
value, a case-ID prefix, or a regex capture group. If no group source is passed,
the output CSV omits `group`. The builder emits only `case_id`, the 16 numeric
features above, and `group` when provided.
