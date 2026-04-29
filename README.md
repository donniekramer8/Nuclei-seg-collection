# Nuclei Segmentation and Feature Extraction

A complete pipeline for segmenting nuclei in H&E whole-slide images using [StarDist](https://github.com/stardist/stardist) and extracting quantitative nuclear morphology features for downstream analysis.

---

## Overview

This repository supports two main workflows:

1. **Train a custom StarDist model** on your own annotated H&E tiles, evaluate it, and apply it to whole-slide images (WSIs).
2. **Use a pretrained H&E model** to immediately segment WSIs and extract nuclear morphology features without any training.

The output of both workflows is a set of per-nucleus measurements (area, shape, color, etc.) saved as pandas DataFrames (`.pkl`) and optionally MATLAB (`.mat`) files for further analysis.

---

## Repository Structure

```
Nuclei-seg-collection/
│
├── nuclei_seg/                  # Core package — import from here
│   ├── __init__.py
│   ├── segmentation.py          # Model loading, WSI segmentation, training utilities, scoring
│   ├── features.py              # Nuclear morphology and color feature extraction
│   └── utils.py                 # Point matching, GeoJSON I/O, contour helpers
│
├── notebooks/                   # Step-by-step Jupyter notebooks
│   ├── 1_train_custom_model/    # Full training workflow (5 notebooks)
│   ├── 2_pretrained_model/      # Quickstart with published H&E model
│   ├── 3_select_nuclei/         # Select nucleus subsets via QuPath annotations
│   ├── 4_cell_seg_from_nuclei/  # Estimate cell boundaries from nucleus positions
│   └── 5_export_geojson/        # Export segmentation results to QuPath GeoJSON
|
├── requirements.txt
└── .gitignore
```

---

## Installation

Python 3.9 is recommended (required by TensorFlow + StarDist).

```bash
git clone <repo-url>
cd Nuclei-seg-collection
pip install -r requirements.txt
```

> **GPU note:** TensorFlow will use a GPU automatically if CUDA drivers are installed. StarDist's `predict_instances_big` is significantly faster on GPU for large WSIs.

---

## Quickstart

### Option A — Use the pretrained H&E model (no training needed)

Open [notebooks/2_pretrained_model/1_segment_and_get_features.ipynb](notebooks/2_pretrained_model/1_segment_and_get_features.ipynb).

This single notebook will:
1. Load StarDist's published `2D_versatile_he` model
2. Segment all WSIs in a directory
3. Extract nuclear morphology features and save them as `.pkl` files

### Option B — Train a custom model on your own data

Work through the notebooks in [notebooks/1_train_custom_model/](notebooks/1_train_custom_model/) in order:

| Notebook | Purpose |
|---|---|
| `1_get_tiles_from_mats.ipynb` | Extract training tiles and masks from `.mat` annotation files |
| `2_StarDist_segment_tiles.ipynb` | Pre-segment tiles with the published model to verify setup |
| `3_Train_Model.ipynb` | Train a custom StarDist model on your annotated tiles |
| `4_Test_Model.ipynb` | Evaluate the model (F1, Precision, Recall, Panoptic Quality) |
| `5_Segment_WSIs.ipynb` | Apply the trained model to all WSIs in a directory |

---

## Workflows in Detail

### Segmentation

```python
from nuclei_seg import load_model, segment_dir_of_images

model = load_model('/path/to/my_model')

segment_dir_of_images(
    WSI_path='/data/slides',
    file_type='.tif',
    out_nm='stardist_output',
    model=model,
    save_tif=False,   # set True to also save label images (~3 GB each)
)
```

Output is written to `/data/slides/stardist_output/json/`, one `.json` file per slide.

**JSON format (one entry per nucleus):**
```json
[
  {
    "centroid": [[row, col]],
    "contour": [[x0, y0], [x1, y1], ...]
  },
  ...
]
```

### Feature Extraction

```python
from nuclei_seg import write_df_features_pkl, write_mat_features_from_pkl

# Extract features for all slides (reads JSON + WSI, writes .pkl)
write_df_features_pkl(
    WSI_path='/data/slides',
    out_name='stardist_output',
    WSI_file_type='.tif',
)

# Optionally convert .pkl files to MATLAB .mat format
write_mat_features_from_pkl(
    WSI_path='/data/slides',
    out_name='stardist_output',
)
```

`.pkl` files are saved to `.../stardist_output/json/nuclear_morph_features_pkl/`.

#### Extracted features (per nucleus)

| Feature | Description |
|---|---|
| `Centroid_x`, `Centroid_y` | Spatial position in the WSI (pixels) |
| `Area` | Nuclear area (px²) |
| `Perimeter` | Contour perimeter (px) |
| `Circularity` | 4π·area / perimeter² — 1.0 = perfect circle |
| `Aspect Ratio` | Major axis / minor axis |
| `compactness` | perimeter² / area |
| `eccentricity` | Ellipse eccentricity (0 = circle, 1 = line) |
| `extent` | area / (major × minor axis) |
| `form_factor` | perimeter² / (4π·area) |
| `maximum_radius` | Max distance from centroid to contour |
| `mean_radius` | Mean distance from centroid to contour |
| `median_radius` | Median distance from centroid to contour |
| `minor_axis_length` | Fitted ellipse minor axis |
| `major_axis_length` | Fitted ellipse major axis |
| `orientation_degrees` | Ellipse orientation angle |
| `r/g/b_mean_intensity` | Mean R, G, B pixel intensity inside nucleus |
| `r/g/b_std` | Std dev of R, G, B intensity inside nucleus |
| `slide_num` | Slide identifier |

### Training Utilities

```python
from nuclei_seg import (
    read_tiles, read_masks, normalize_images,
    augment_tiles, split_train_val_set,
)

tiles = read_tiles('/data/tiles/HE')
masks = read_masks('/data/tiles/masks')
tiles_norm = normalize_images(tiles)

# 8x augmentation (rotations + horizontal flip)
tiles_aug, masks_aug = augment_tiles(tiles_norm, masks)

tiles_train, masks_train, tiles_val, masks_val = split_train_val_set(
    tiles_aug, masks_aug, val_ratio=0.15
)
```

### Model Evaluation

```python
from nuclei_seg import get_stats

taus = [0.5, 0.6, 0.7, 0.8]
results = get_stats('/data/tiles/HE', masks_gt, masks_pred, taus)
print(results)
```

Metrics computed per tile per IoU threshold (tau):

- **IoU** — pixel-level intersection over union
- **TP / FP / FN** — centroid-based object detection counts
- **Precision / Recall / F1**
- **Segmentation Quality (SQ)** — mean IoU of matched objects
- **Panoptic Quality (PQ)** — SQ × F1

### QuPath Export

```python
from nuclei_seg import json_to_geojson_whole_folder

json_to_geojson_whole_folder('/data/slides/stardist_output/json')
# GeoJSON files are written to .../json/geojson/
```

Load the resulting `.geojson` files in QuPath via **Objects → Import objects → GeoJSON**.

### Point Colocalization

Match nucleus centroids between two sets (e.g. QuPath manual annotations vs. model predictions):

```python
from nuclei_seg import get_json_centroids, colocalize_points

centroids_a, _ = get_json_centroids('/data/pred.json')
centroids_b, _ = get_json_centroids('/data/annot.json')

row_match, col_match = colocalize_points(centroids_a, centroids_b, r=20)
```

---

## Data Format Notes

- **Input images:** `.tif` or `.png` RGB H&E images. Large WSIs are processed in 4096×4096 blocks with 128 px overlap via `predict_instances_big`.
- **Normalization:** Images are divided by 255 before inference. Models were trained on data normalized this way.
- **Coordinate convention:** StarDist returns coordinates as `(row, col)` i.e. `(y, x)`. The JSON output stores centroids as `[row, col]` and contours as `[[x, y], ...]` after flipping. Keep this in mind when integrating with other tools.

---

## Dependencies

| Package | Purpose |
|---|---|
| `tensorflow` | StarDist model training and inference backend |
| `stardist` | Star-convex polygon segmentation |
| `opencv-python` | Contour analysis, mask drawing |
| `scikit-learn` | Nearest-neighbor search for point matching |
| `scipy` | Bipartite matching, `.mat` file I/O |
| `pandas` | Feature DataFrame storage |
| `numpy` | Array operations |
| `matplotlib` | Visualization |
| `geojson` | QuPath-compatible export |
| `tifffile` | TIFF image I/O |
| `imagecodecs` | Compressed TIFF codec support |
| `Pillow` | Image augmentation |
| `h5py` | HDF5 model weight files |
| `tqdm` | Progress bars |
