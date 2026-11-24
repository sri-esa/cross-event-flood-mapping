# Reproducibility Guide

## 1. Environment

All training, evaluation, and inference experiments were conducted in Google Colab (NVIDIA Tesla T4 GPU for training; GPU or CPU supported for evaluation and inference).

Core model stack:

- Python 3 / PyTorch
- `segmentation-models-pytorch==0.6.0`
- `rasterio`, `scikit-image`, `numpy`, `pandas`, `matplotlib`
- Architecture: U-Net with an **ImageNet-pretrained ResNet-34** encoder (`~24.43M` trainable parameters)

See the root-level `requirements.txt` for the complete dependency list.

## 2. Datasets

### A. Sen1Floods11 Hand-Labeled Dataset (v1.1)

Contains 2-channel Sentinel-1 SAR imagery (`VV`, `VH` in dB; `512 × 512` pixels) with pixel-level masks (`-1` = ignore/no-data, `0` = non-water, `1` = water). Six unusable chips were excluded (5 completely unlabelled all-`-1` chips and 1 all-NaN SAR chip, `Paraguay_34417_S1Hand.tif`):

| Split | Usable Images | Role |
|---|---:|---|
| Training (`train_clean.csv`) | 251 | Model training & normalization statistics |
| Validation (`valid_clean.csv`) | 86 | Early stopping & best-checkpoint selection |
| Testing (`test_evaluable.csv`) | 88 | Unseen in-distribution test evaluation (E0, E1) |
| Bolivia (`flood_bolivia_data.csv`) | 15 | Held-out geographic transfer evaluation (E0, E1) |
| **Total** | **440** | |

### B. Bengal Delta — Cyclone Amphan (May 2020)

- **Area of Interest (AOI H):** `[88.4, 21.75, 88.9, 22.25]` (`~2,859 km²`, `data/aoi.geojson`).
- **Sentinel-1 SAR Scenes:** Descending, relative orbit 150 (pre-event: `16 May 2020`; post-event: `28 May 2020`).
- **Benchmark Chips:** 40 chips (`256 × 256` at `10 m`, `EPSG:32645`) across four regimes (`20` flood-rich, `10` ambiguous, `6` negative-dry, `4` negative-water) with QGIS-verified final labels (`0` = dry, `1` = flood, `-1` = ignore).
- **Leak-Free Spatial Split (E3):** **NW + NE (`27` chips)** used as the fine-tuning pool (`N = 5, 10, 20`); **SW + SE (`13` chips)** held out as the fixed spatial test set.

## 3. Preprocessing & Model Training

### Preprocessing (`normalization_stats_clipped.json`)

Computed strictly from the `251` cleaned training images (1st and 99th percentiles, population standard deviation):

| Channel | Clip Min (dB) | Clip Max (dB) | Mean (dB) | Population Std (dB) |
|---|---:|---:|---:|---:|
| Band 0 — VV | -23.8216 | -2.4397 | -10.399012 | 3.880121 |
| Band 1 — VH | -34.0875 | -9.5387 | -17.217060 | 4.482825 |

Invalid SAR pixels and `-1` label pixels are masked out; invalid SAR pixels are zeroed after standardization. Training uses synchronized random `256 × 256` crops, horizontal/vertical flips, and random `90°` rotations; validation and test run unaugmented at full resolution.

### Training Configuration

- Architecture: U-Net with ImageNet-pretrained ResNet-34 encoder
- Input / Output: 2 channels (`VV`, `VH`) $\rightarrow$ 1 logit channel (sigmoid + `0.5` threshold at inference)
- Loss: Masked `0.5 × BCE + 0.5 × Dice` (excluding `-1` / invalid pixels)
- Optimizer: Adam (`lr = 0.0001`), batch size `16` (train) / `2` (eval), mixed precision (`float16`)
- Scheduler: `ReduceLROnPlateau` monitoring validation flood IoU (`factor = 0.5`, `patience = 2`)
- **Maximum epochs:** `40` (early stopping `patience = 7`, `min_delta = 0.0001` stopped training at **Epoch 29**)
- **Best frozen checkpoint:** **Epoch 22** (`best_unet_resnet34_final_iou.pth`, validation IoU `0.63866`)

## 4. Final Evaluation Summary

All experiments use the shared `-1`-safe evaluation module (`metrics.py`). Complete results are recorded in `results/results_table.csv`:

| Experiment | Test Set | Status | IoU | Precision | Recall | F1 | GT Area (km²) | Pred Area (km²) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| E0 — Otsu (VH) | Sen1Floods11 Test (88) | Final | 0.23640 | 0.25627 | 0.75297 | 0.38240 | 251.6075 | 739.2644 |
| E1 — Frozen U-Net | Sen1Floods11 Test (88) | Final | 0.63605 | 0.79409 | 0.76168 | 0.77755 | 251.5008 | 241.2355 |
| E0 — Otsu (VH) | Sen1Floods11 Bolivia (15) | Final | 0.38445 | 0.40637 | 0.87698 | 0.55538 | 45.1901 | 97.5241 |
| E1 — Frozen U-Net | Sen1Floods11 Bolivia (15) | Final | 0.62142 | 0.83452 | 0.70876 | 0.76652 | 45.1901 | 38.3799 |
| E0 — Otsu (VH) | Bengal Spatial Test (13) | Final (v2) | 0.15730 | 0.16768 | 0.71752 | 0.27183 | 6.3555 | 27.1958 |
| E2 — Zero-Shot U-Net | Bengal Spatial Test (13) | Final (v2) | 0.12085 | 0.16622 | 0.30687 | 0.21564 | 6.3555 | 11.7330 |
| E3 — Fine-Tuned (`N = 5`) | Bengal Spatial Test (13) | Final (v2) | 0.20135 | 0.27999 | 0.41753 | 0.33520 | 6.3555 | 9.4774 |
| E3 — Fine-Tuned (`N = 10`) | Bengal Spatial Test (13) | Final (v2) | 0.21984 | 0.27795 | 0.51256 | 0.36044 | 6.3555 | 11.7202 |
| E3 — Fine-Tuned (`N = 20`) | Bengal Spatial Test (13) | Final (v2) | 0.23707 | 0.27999 | 0.60735 | 0.38328 | 6.3555 | 13.7864 |

## 5. Initial Amphan Practice-Window Inference (Pre/Post Water Transition)

Before the 40-chip AOI H benchmark, the frozen checkpoint was run over a `~117 km²` (`114.66 km²` valid pixels; `1120 × 1047` px at `10 m`, `EPSG:32645`) pre/post practice sub-window using `512 × 512` sliding windows (`stride = 384`) with 2D Hanning centre-weighted blending:

| Measurement (`~117 km²` Practice Window) | Predicted Area |
|---|---:|
| Pre-event predicted water | 18.8736 km² |
| Post-event predicted water | 17.9362 km² |
| Persistent predicted water | 17.7785 km² |
| Newly predicted water | 0.1577 km² |
| Water no longer predicted | 1.0951 km² |
| Net predicted change | -0.9374 km² |
| Pre/post water-map Jaccard overlap | 0.93417 |

**Important caveat:** These values cover only the `~117 km²` practice sub-window (not the full `2,859 km²` AOI H) and measure pre/post predicted surface-water change rather than validated flood extent. They should not be compared directly against the `58.0 km²` AOI H v1 flood total; use the 40-chip benchmark (`results/results_table.csv`) for apples-to-apples flood evaluation.

## 6. Storage Layout (GitHub vs. Google Drive)

- **GitHub (lightweight code, tables, figures, docs):** `metrics.py`, `requirements.txt`, `data/aoi.geojson`, `notebooks/`, `results/*.csv`, `results/*.png`, and `report/REPRODUCIBILITY.md`.
- **Shared Google Drive (`flood_mapping/Bengal_Flood_Project/`):** Raw imagery and chips (`Chips/`), QGIS hand-correction files (`QGIS_Work/`), rasterized final and review labels (`labels_final/`, `labels_review/`), frozen checkpoints and logs (`Models/`), and prediction GeoTIFFs (`Results/`).

## 7. Reproduction Instructions

1. Ensure the shared `flood_mapping` Drive folder is accessible in `My Drive` (via a Drive shortcut if opened from `Shared with me`).
2. **To reproduce Sen1Floods11 test evaluation & Amphan window inference (no retraining):** Open `notebooks/flood_mapping_reproduction.ipynb` in Google Colab, mount Drive, install `requirements.txt` (`segmentation-models-pytorch==0.6.0`, `rasterio`), verify paths point to `Models/checkpoints/best_unet_resnet34_final_iou.pth` and `Models/logs/normalization_stats_clipped.json`, and run the evaluation/inference sections.
3. **To reproduce E0–E3 Bengal & Bolivia tables and figures:** Open `notebooks/member_c_experiments.ipynb` in Google Colab, ensure `metrics.py` is available, verify paths point to `Bengal_Flood_Project/Chips/`, `labels_final/`, and `Models/`, and run all cells.
