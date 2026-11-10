# Reproducing STAC from data to selected cases

STAC's selector requires one feature row per candidate training case. The
included workflow uses a full-data nnU-Net warm-up checkpoint at epoch 40 to
produce the model-dependent features. Data, pretrained weights, and prediction
caches are intentionally not bundled.

## 1. Obtain and prepare data

- **LoveDA:** download the original dataset from its [official project
  page](https://github.com/Junjue-Wang/LoveDA), which links the dataset
  release at [Zenodo](https://doi.org/10.5281/zenodo.5706578). Follow the
  dataset's terms and cite its paper.
- **REDD Additional:** this project-specific dataset is not redistributed
  here. Obtain the exact release and access permission identified by the STAC
  paper's data-availability statement. Do not upload source imagery or labels
  unless their owner explicitly permits redistribution.

Convert the selected dataset to the nnU-Net raw format. Use matching case IDs
for images and masks:

```text
$nnUNet_raw/DatasetXXX_STAC/
  dataset.json
  splits_final.json
  imagesTr/<case>_0000.png
  labelsTr/<case>.png
  imagesTs/<case>_0000.png       # optional for selection; needed for evaluation
  labelsTs/<case>.png            # optional for selection; needed for evaluation
```

Prepare newline-delimited `train_ids.txt` and `val_ids.txt` using the exact
paper split. Put source RGB tiles and semantic masks in separate directories;
the helper pairs files by stem, including nested source folders:

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
export DATASET_ID=501
export DATASET_NAME=Dataset501_STAC

python scripts/prepare_nnunet_dataset.py \
  --images /data/source/images \
  --labels /data/source/labels \
  --train-ids /data/splits/train_ids.txt \
  --val-ids /data/splits/val_ids.txt \
  --output-root "$nnUNet_raw" \
  --dataset-name "$DATASET_NAME" \
  --foreground-value 6 \
  --test-images /data/source/test_images \
  --test-labels /data/source/test_labels
```

Omit the two test flags if test labels are unavailable or unnecessary. The
script converts images to RGB PNG and masks to binary PNG (`0` background,
`1` target), writes `dataset.json` and `splits_final.json`, and refuses to
overwrite an existing dataset. For the paper's binary LoveDA setup, source
class 6 is forest (`1`) and all other classes are background (`0`). For REDD,
set `--foreground-value` to the positive class value in the authorized binary
mask. Do not derive or tune the split using the test set.

Example `dataset.json` for RGB PNGs:

```json
{
  "channel_names": {"0": "R", "1": "G", "2": "B"},
  "labels": {"background": 0, "forest": 1},
  "numTraining": 100,
  "file_ending": ".png",
  "name": "DatasetXXX_STAC",
  "overwrite_image_reader_writer": "NaturalImage2DIO"
}
```

Replace `100` with the number of training cases. Assign an unused numeric dataset ID and retain it in all commands below. The
following uses `501` as an example; change both the directory name and ID if it
is already in use.

Install the base selector and its feature-builder dependencies:

```bash
python -m pip install -e '.[features]'
```

Install a compatible nnU-Net version separately, then plan and preprocess:

```bash
nnUNetv2_plan_and_preprocess -d "$DATASET_ID" --verify_dataset_integrity
```

## 2. Train or provide the epoch-40 warm-up checkpoint

The paper uses a full-data warm-up model to generate prediction features. To
train that model with nnU-Net, install the included trainer extension by copying
it into the installed nnU-Net trainer package:

```bash
NNUNET_TRAINER_DIR="$(python -c 'import nnunetv2.training.nnUNetTrainer as p; print(p.__path__[0])')"
cp extensions/nnUNetTrainer_STAC.py \
  "$NNUNET_TRAINER_DIR/variants/training_length/nnUNetTrainer_STAC.py"
nnUNetv2_train "$DATASET_ID" 2d 0 \
  -tr nnUNetTrainer_STAC_400epochs_Save40 \
  -p nnUNetPlans
```

Alternatively, use the exact full-data warm-up checkpoint and trainer version
used by the experiment. Record the nnU-Net version/commit, checkpoint hash,
split hash, and command in the experiment metadata. The expected checkpoint is
`checkpoint_epoch_0040.pth` in the model's `fold_0` directory.

## 3. Export warm-up probabilities and build features

Run inference on the `imagesTr` directory; the feature builder below explicitly
filters to the split's training candidates and does not include validation or
test cases. The `--save_probabilities` output is consumed by the converter.
Export candidate IDs from the official split:

```bash
PRED_DIR="$PWD/work/predictions"
ARTIFACT_DIR="$PWD/work/artifacts"
mkdir -p work
python - "$nnUNet_raw/$DATASET_NAME/splits_final.json" > work/candidate_case_ids.json <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    split = json.load(handle)[0]
json.dump(split["train"], sys.stdout)
PY

nnUNetv2_predict \
  -i "$nnUNet_raw/$DATASET_NAME/imagesTr" \
  -o "$PRED_DIR" \
  -d "$DATASET_ID" \
  -c 2d \
  -f 0 \
  -tr nnUNetTrainer_STAC_400epochs_Save40 \
  -p nnUNetPlans \
  -chk checkpoint_epoch_0040.pth \
  --save_probabilities

python scripts/convert_nnunet_predictions.py \
  --prediction-dir "$PRED_DIR" \
  --output-dir "$ARTIFACT_DIR" \
  --foreground-class 1 \
  --threshold 0.5
```

Build the feature table from the original training images and binary labels.
Set the group source to the experiment's metadata strata: for LoveDA, `Rural`
or `Urban`; for REDD, use the region prefix of each case ID.

```bash
# LoveDA
python scripts/build_features.py \
  --image-dir "$nnUNet_raw/$DATASET_NAME/imagesTr" \
  --label-dir "$nnUNet_raw/$DATASET_NAME/labelsTr" \
  --probability-dir "$ARTIFACT_DIR/probabilities" \
  --hard-mask-dir "$ARTIFACT_DIR/hard_masks" \
  --case-ids work/candidate_case_ids.json \
  --group-regex '^(?:LoveDA_)?(?P<group>Rural|Urban)_' \
  --output work/features.csv

# REDD instead: use --group-from-prefix and write to a different feature file.
```

If filenames do not encode the group, provide `--metadata-csv` with `case_id`
and `group` columns instead. The feature schema and definitions are in
[FEATURE_SCHEMA.md](FEATURE_SCHEMA.md).

## 4. Select the subset

```bash
select-method-v3 select \
  --features work/features.csv \
  --retention 0.10 \
  --representation-mode group_equal_with_exg \
  --quota-mode gt_metadata_adaptive_legacyfill \
  --gt-background-scale 0.685 \
  --gt-value-weight 0.10 \
  --forest-bonus-weight 0 \
  --forest-bonus-cap 0 \
  --guard-swap-ratio 0 \
  --output-dir work/selection
```

The final selected case IDs are in `work/selection/selection.json`; the run
configuration, input hash, strata, and quotas are recorded in
`work/selection/summary.json`. The selection output is the end of this
repository's pipeline; downstream training/evaluation should use the same
published split and training protocol as the paper.

## Reproducibility limits

The exact REDD download/access URL, paper-specific split files, and checkpoint
artifacts must be supplied or cited by the authors before claiming a
bit-for-bit reproduction. Dataset and checkpoint licenses remain in force
independently of this code repository.
