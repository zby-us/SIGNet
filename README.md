# SIGNet

PyTorch implementation of **SIGNet: A Hierarchical Deep Learning Framework for CT-Based Sacroiliitis Grading**.

SIGNet segments four sacroiliac structures, constructs bilateral ROIs and structural priors, and predicts slice-level and case-level grades.

- [Segmentation: inputs, outputs, training and pretrained weights](segment/README.md)
- [Grading: inputs, outputs and training](model/README.md)
- [Data preparation and method details](docs/USAGE.md)

## Structure

```text
SIGNet/
├── segment/       # Segmentation code, examples and weight link
├── model/         # Grading code, example inputs and label tables
├── main.py        # Two-stage inference
├── utils.py
├── requirements.txt
├── docs/          # Detailed usage
└── tests/         # Input and checkpoint checks
```

## Installation

Python 3.10 or later.

```bash
python -m pip install -r requirements.txt
```

## Complete inference

Run from the repository root:

```bash
python main.py --input-dir /path/to/images --slice-manifest /path/to/slice_manifest.csv --segmentation-checkpoints /path/to/segmentation_weights --grading-checkpoint-left /path/to/dualfuse_band_left_best.pth --grading-checkpoint-right /path/to/dualfuse_band_right_best.pth --output-dir outputs/result
```

## Citation

Please cite **SIGNet: A Hierarchical Deep Learning Framework for CT-Based Sacroiliitis Grading** when using this work. Full bibliographic details will be added when available.

This code is intended for research. See [validation status](docs/USAGE.md#validation-status).
