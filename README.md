# STAC

Code for the IEEE BigData paper **“STAC: STrAtified Coverage for Extreme-Budget Coreset Selection in Remote-Sensing Segmentation.”**

STAC selects a representative training subset from segmentation features. Dataset files, trained weights, and prediction caches are not included. See [docs/REPRODUCING.md](docs/REPRODUCING.md) for the data-to-selection workflow.

```bash
python -m pip install -e '.[features,test]'
pytest
```

**License:** Not yet specified by the authors. Until a license is added, no permission to reuse, modify, or redistribute this code is granted. Dataset and checkpoint terms are separate.
