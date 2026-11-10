# Model integration

The selector is model-agnostic. An integration has one responsibility: produce
one feature CSV row for every candidate training case.

## Adapter flow

~~~text
checkpoint + training images
          |
          v
warm-up prediction/probability export
          |
          v
feature extraction -> artifact_features.csv
          |
          v
select-method-v3 -> selected case IDs
~~~

The model adapter may be implemented for nnU-Net, PyTorch, MONAI, or another
framework. It must preserve the feature definitions in
`FEATURE_SCHEMA.md`. In particular, prediction masks must use the same class
mapping and threshold, and probability features must be computed before any
lossy mask export.

The selector consumes exactly one artifact table per run. It never reads a
second artifact, checkpoint, prediction directory, dataset ID, or model
configuration. This boundary makes the method reusable and makes input hashes
in `summary.json` sufficient to identify the selection inputs.

