# Data and training

Run all commands from the repository root. Replace `/path/to/...` with your own paths.

## Segmentation data

Each `train/` and `val/` split contains:

```text
images/<case>/<slice>.png
masks_4c/left/ilium/<case>/<slice>_left_ilium.png      # LI
masks_4c/left/sacrum/<case>/<slice>_left_sacrum.png    # LS
masks_4c/right/ilium/<case>/<slice>_right_ilium.png # RI
masks_4c/right/sacrum/<case>/<slice>_right_sacrum.png    # RS
```

Case subdirectories are optional, but image and mask relative paths must match.

LI means left ilium, LS left sacrum, RI right ilium, and RS right sacrum.
`STRUCTURES` stores `(stored-image half, anatomical side, bone)`:

| Structure | Anatomical name | Stored-image half | Mask directory |
|---|---|---|---|
| LI | Left ilium | left | `masks_4c/left/ilium/` |
| LS | Left sacrum | left | `masks_4c/left/sacrum/` |
| RI | Right ilium | right | `masks_4c/right/ilium/` |
| RS | Right sacrum | right | `masks_4c/right/sacrum/` |

Mask directories and filename suffixes use anatomical side and bone names. Prepare both the target bone and its same-side companion mask using the paths above. Preserve the supplied image orientation; displayed image left/right alone does not establish patient laterality.


```bash
python -m segment.train --data-root /path/to/segmentation_data --output-dir outputs/segment_weights
```

Defaults: 30 epochs, AdamW, learning rate 2e-4, batch size 2, four-step gradient accumulation. Each structure has a validation-selected threshold searched over 0.01–0.99 in steps of 0.01. Use the selected best30 checkpoints for inference.

## Grading data

Each `train/` and `val/` split contains `labels.csv`, `roi/` and `roi_masks/`:

```text
labels.csv                         # filename,left,right
roi/<stem>_left.png
roi/<stem>_right.png
roi_masks/<stem>_left_mask.png
roi_masks/<stem>_right_mask.png
```

The paper reports 2,452 training and 538 validation slices for Dataset A. The training and validation label tables are included under `model/data/Dataset_A/`. Full image datasets are not included; one author-supplied paper example and its derived ROI inputs are provided under `segment/examples/` and `model/examples/`. Prepare image and mask directories following the layout above. Preserve the relative filename layout when preparing inputs. Validation and test ROIs must come from predicted segmentation masks. `segment.predict` generates the paired images and provenance file `inference_metadata.json`.

The grading training entry checks validation provenance when this file exists; legacy inputs without it emit a warning and remain supported. This checks a declaration, not the contents' actual provenance. Missing ROI images, masks, or labels stop training instead of silently reducing the dataset. Both sides are checked before training. SWA parameters are averaged during training but are not exported as a separate checkpoint.

```bash
python -m model.train --data-root /path/to/grading_data --output-dir outputs/grading_weights
```

This is the classification training entry point. It includes image augmentations, mask perturbations, MixUp, CutMix, EMA and SWA. Defaults: 100 epochs without early stopping, batch size 32, AdamW, learning rate 2e-4, weight decay 1e-4, five warmup epochs. SWA parameter averaging and its learning-rate schedule start at epoch 50. The best EMA checkpoint is selected by validation accuracy and used for testing; final raw parameters are saved separately.

Training reports are per-side validation diagnostics. The paper's pooled left/right test-set metrics are a separate evaluation; these training reports do not reproduce the paper's result tables.

## Slice manifest

Case inference requires an explicit ordering manifest, separate from grading labels:

```csv
patient_id,case_id,slice_id,slice_order,image_path
p001,1,74,74,1/img-00003-00074.png
p001,1,75,75,1/img-00003-00075.png
```

The manifest lists original image paths relative to the input root. Each entry requires both left and right ROI/mask pairs. Missing files cause an error before grading weights are loaded. Slice order gaps break consecutive runs.

## Weights and naming

Segmentation checkpoint names use `unet_vmamba_best30_LI.pt`, `LS.pt`, `RI.pt`, and `RS.pt`. Each checkpoint must correspond to its own structure. The loader checks `structure` metadata when present and otherwise uses the anatomical filename; `side` metadata is checked when available. Saved validation thresholds are read automatically; legacy parameter-only checkpoints require explicit validation-selected `--thresholds` values.

See [segmentation weights](../segment/README.md). Keep the original filenames in one local folder.

| File | Structure | Threshold | SHA256 |
|---|---|---|---|
| unet_vmamba_best30_LI.pt | LI | 0.95 | 455ee99582922c4654134430652cf760da4f708fd5d3c81974b9e018ce6d6d0a |
| unet_vmamba_best30_LS.pt | LS | 0.93 | 1bd4fa3157b828363e024b609d09b5c871843b41a5d317a8c0b149618cf4439b |
| unet_vmamba_best30_RS.pt | RS | 0.91 | 9f766de18c0f6b8a297cbcac87839dc175238d626e38c42a0705e5907cd2b44f |
| unet_vmamba_best30_RI.pt | RI | 0.99 | 64c5fda6e0b9140c2189f4a036a417bf3e72cf47d4b5517623cdab5920e91511 |

All four checkpoints record threshold selection on Dataset A validation data, using 0.01–0.99 in steps of 0.01.

The grading training entry saves `dualfuse_band_left_best.pth` and `dualfuse_band_right_best.pth`. Set the inference checkpoint paths to the corresponding local files.

## Preprocessing

Generate the SDT and the boundary band at the original ROI size. The band uses a 3x3 morphological gradient and a 7x7 elliptical dilation. Preserve uint8 SDT quantization, stack ROI/mask/SDT/band, linearly resize all channels together to 224x224, then normalize. No fixed CLAHE or contrast multiplier is applied. Inference averages original and horizontally flipped logits.

## Validation status

Local checks cover selected components and input validation. All four supplied segmentation checkpoints passed strict parameter loading, finite-value checks and a 512x256 forward pass. A single CT slice has also been processed with the final four segmentation checkpoints. The displayed paper example separately uses author-supplied masks; its ROI images and ROI masks are derived from those supplied masks and checked for alignment. These supplied masks are not presented as outputs from that checkpoint test. Full training and complete real-data grading evaluation have not been verified. Exact historical augmentation-library versions are unavailable; dependency compatibility remains to be checked. This repository does not claim verified numerical reproduction of all paper tables.
