# VLM Teacher–RNN Student: Time-Series Anomaly Detection
### IE643 Course Project — Knowledge Distillation for Anomaly Detection

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-green.svg)](https://streamlit.io/)

---

## Overview

This application implements a **Vision-Language Model (VLM) guided Knowledge Distillation** framework for time-series anomaly detection. A frozen CLIP model acts as the "teacher," generating candidate anomaly scores by comparing heatmap visualizations of time-series windows against text prompts describing normal and anomalous behaviour. A lightweight GRU-based "student" model is then trained through a combination of:

- **Supervised learning** (when ground-truth labels are available)
- **Knowledge distillation** (learning from teacher's soft probability distributions)
- **Time-step localization** (pinpointing which time steps are anomalous)

> **Scientific note:** CLIP was not designed for time-series anomaly detection. Its image–text similarity scores are *candidate teacher signals*, not calibrated anomaly probabilities. The application evaluates teacher scores against validation labels and warns the user if performance is poor.

> **Project status:** Five dataset adapters are implemented. Local checkpoints and measured results are present for some datasets under `checkpoints/`; each result is scoped to its saved validation/test split and should not be presented as completion of every dataset. See [ANOMALY_METHOD.md](ANOMALY_METHOD.md) for the point/contextual/collective decision logic and CreditCard limitations, and [PROJECT_WALKTHROUGH.md](PROJECT_WALKTHROUGH.md) for the MRI paper explanation, requirement-by-requirement status, dataset caveats, and the experiment sequence.

---

## v2 (post primary-assessment revision) — start here

The revision answers the primary-assessment review with a controlled benchmark.
See [METHODOLOGY.md](METHODOLOGY.md) for the exact training objective (which
target trains which head, with a worked example) and a point-by-point response
to the review. Results are in `experiments/benchmark_v2/RESULTS.md`.

| Step | Command |
|---|---|
| Run every model on all five datasets (identical splits, seeds 0/1/2) | `.venv\Scripts\python.exe scripts\run_benchmark.py` |
| One dataset / one seed (quick) | `.venv\Scripts\python.exe scripts\run_benchmark.py --datasets daphnet --seeds 0` |
| Summaries, teacher export, demo CSVs, figures | `.venv\Scripts\python.exe scripts\finalize_benchmark.py` |
| Independent metric re-computation + Daphnet v1→v2 attribution | `.venv\Scripts\python.exe scripts\verify_results.py` |
| Label-free out-of-sample suite (all datasets) | `.venv\Scripts\python.exe scripts\run_oos_suite.py` then `scripts\plot_oos.py` |
| One label-free test CSV + hidden answer key | `.venv\Scripts\python.exe scripts\generate_unlabeled_test.py --dataset metropt` |
| Score any CSV with teacher and student | `.venv\Scripts\python.exe scripts\score_unlabeled.py --dataset metropt --csv <file> [--answer-key <key>]` |
| Interface (8 pages: data pipeline, teacher, student & training, evaluation step by step, teacher vs student, results, metrics guide) | `streamlit run app.py` |
| Tests | `.venv\Scripts\python.exe -m pytest -q tests` |

Scope: Credit Card, Daphnet FoG, 3W and MetroPT-3 (real labels). Air Quality
(synthetic labels) is excluded for now; its runs are in `experiments/excluded_airquality/`.
Results: `experiments/benchmark_v2/RESULTS.md`, `VERIFICATION.md`, `OUT_OF_SAMPLE.md`.

What changed in v2, briefly:

* **Teacher.** CLIP ViT-B/32 stays frozen. Windows are rendered with
  per-channel clipping (the v1 global colour range made most windows
  near-uniform images). Two teachers are compared: prompt-ensemble zero-shot
  CLIP, and a cross-fitted linear probe on frozen CLIP image embeddings
  (`src/teacher/probe.py`).
* **Student.** The GRU gains a small convolutional front-end, bidirectional
  context and mean+max pooling (`src/student/model.py`), plus a training-only
  head for CLIP feature distillation.
* **Objective.** `L = L_step + β·L_window + λ_KD·T²·KL + λ_feat·(1 − cos)`
  (`src/student/losses.py`). Models are selected on validation AUPRC.
* **Localization.** Window-probability gating, smoothing, threshold and
  minimum event length are tuned on validation (`src/evaluation/postprocess.py`).
* **Ablations and baselines.** Zero-shot CLIP, probe CLIP, Isolation Forest,
  gradient-boosted trees, a GRU on ground truth only, KD from each teacher,
  and teacher-only supervision — all on the same test windows.

Deployable v2 checkpoints are written to `checkpoints/v2_<dataset>_gru_kd_full`
(distilled student) and `checkpoints/v2_<dataset>_gru_gt` (labels-only GRU).
They can also be activated on page 5 and used on page 7.

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate synthetic data (for testing)
```bash
python scripts/generate_synthetic.py --output-dir ./data/synthetic
```

### 3. Launch the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## What Input Does the App Accept?

### Custom CSV Upload (Page 2: Upload & Inspect)

The app accepts **any CSV file** containing time-series data. You do not need to rename columns — the app lets you select which columns are which.

#### Supported Formats

| Format | Description | Example |
|--------|-------------|---------|
| **Univariate** | Timestamp + one numeric feature | `timestamp, temperature, label` |
| **Multivariate** | Timestamp + multiple features | `timestamp, V1, V2, ..., V28, Amount, Class` |
| **No timestamp** | Row order used as time index | `feat1, feat2, label` |
| **No labels** | Time series without anomaly annotations | `timestamp, sensor1, sensor2` |
| **Long format** | Timestamp + variable + value | Pivot to one numeric column per feature before upload |
| **Multi-entity** | Multiple independent series in one file | With a group/entity ID column |

#### Column Types

After uploading, you will be asked to select:

| Role | Description | Required? |
|------|-------------|-----------|
| **Timestamp** | Date/time column for chronological ordering | Optional (row order used if absent) |
| **Feature columns** | Numeric measurement columns to model | **Required** (at least 1) |
| **Label column** | Ground-truth anomaly indicator | Optional |
| **Normal label value** | What value means "normal" in the label column | Required if label selected |
| **Anomaly label value** | What value means "anomalous" in the label column | Required if label selected |
| **Entity/Series ID** | Column identifying independent sub-series | Optional |

#### Common Issues

| Problem | Solution |
|---------|----------|
| Non-numeric columns included | The app will detect and exclude them. You can override. |
| Missing values (`NaN`, `-200`) | Choose interpolation / forward-fill / mean in Preprocessing. |
| Timestamps in unusual format | Use pandas `parse_dates` auto-detection or provide ISO 8601 format. |
| Mixed-frequency data | Resample to a fixed frequency (supported for MetroPT). |

---

## Built-in Datasets

You can load any of the five pre-configured datasets from the **Upload & Inspect** page without uploading a CSV yourself. Download the data first using the scripts below.

### Dataset 1: Credit Card Fraud (Kaggle)

- **Source:** [Kaggle mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **File:** `data/creditcard.csv`
- **Download:** Manual download from Kaggle (free account required)
- **Features:** V1–V28 (PCA components) + Amount (F=29)
- **Labels:** `Class` (0=normal, 1=fraud)
- **Anomaly rate:** ~0.17%
- **Window:** L=64, stride=32
- **Split:** Chronological 60/20/20
- **Headline metric:** AUPRC (not accuracy — severe class imbalance)
- **⚠️ Note:** Consecutive rows represent different cardholders, so temporal structure is weak. Anomalies are isolated point events.

### Dataset 2: Daphnet Freezing of Gait (UCI)

- **Source:** [UCI id 245](https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait)
- **Download:**
  ```bash
  python scripts/download_daphnet.py
  ```
- **File:** `data/daphnet/` (multiple `.txt` files, one per recording)
- **Features:** 9 accelerometer channels (ankle, thigh, trunk × x/y/z) (F=9)
- **Labels:** `annotation` (1=normal, 2=freeze/anomaly; 0=not in experiment → dropped)
- **Sampling:** 64 Hz
- **Window:** L=256 (4 seconds), stride=64
- **Split:** By subject (so no subject appears in both train and test)
- **Note:** Gaps at `annotation=0` rows create recording segment boundaries. Windows never cross these boundaries.

### Dataset 3: UCI Air Quality

- **Source:** [UCI id 360](https://archive.ics.uci.edu/dataset/360/air+quality)
- **Download:**
  ```bash
  python scripts/download_airquality.py
  ```
- **File:** `data/AirQualityUCI.csv`
- **Features:** 13 sensor/reference columns; NMHC(GT) dropped (>50% missing) (F=12)
- **Labels:** ⚠️ **None in original data.** Choose a label strategy:
  - `semi_synthetic` (default): Point spikes, level shifts, drift, and noise bursts injected independently into train/validation/test after splitting. Magnitudes use clean training-set scales; all metrics are tagged **"SYNTHETIC ANOMALIES."**
  - `dropout`: Long runs of sensor dropout (-200 codes) labelled as anomaly.
- **Missing values:** -200 → NaN; retained features are imputed after splitting, then the scaler is fit on training data only
- **Window:** L=48 (≈2 days at hourly resolution), stride=12
- **⚠️ WARNING:** Any metric computed on this dataset reflects injected anomalies only and cannot be treated as a real anomaly detection benchmark.

### Dataset 4: 3W Oil-Well Dataset

- **Source:** [GitHub — ricardovvargas/3w_dataset](https://github.com/ricardovvargas/3w_dataset)
- **Download:**
  ```bash
  python scripts/download_3w.py
  ```
  (Provides instructions + optionally runs `git clone`)
- **File:** `data/3w/` (many subdirectories, ~2000 CSV files)
- **Features:** P-PDG, P-TPT, T-TPT, P-MON-CKP, T-JUS-CKP, P-JUS-CKGL, T-JUS-CKGL, QGL (F=8)
- **Labels:** `class` (0=normal, nonzero=undesirable event; transients 101–108 counted as anomalous by default)
- **Window:** L=120, stride=60 (per file; no cross-file windowing)
- **Split:** By instance (stratified, so each split has anomalous instances)
- **Missing values:** Treat NaN, +/-inf, and values outside float32 range as missing; forward-fill then backward-fill within each well instance. An entirely missing sensor is set to zero (optional NaN flags are off by default).
- **Scale:** 50.9 million timesteps in the supplied archive; training windows default to a deterministic 20,000-window sample to fit memory, while validation and test windows remain complete. Training-time validation uses up to 5,000 windows; threshold selection and final test metrics use the full held-out partitions.

### Dataset 5: MetroPT-3 Air Compressor

- **Source:** [UCI id 791](https://archive.ics.uci.edu/dataset/791/metropt+3+dataset)
- **Download:**
  ```bash
  python scripts/download_metropt.py
  ```
- **File:** `data/MetroPT3(AirCompressor).csv`
- **Features:** 7 analog (TP2, TP3, H1, DV_pressure, Reservoirs, Motor_current, Oil_temperature) + 8 digital (COMP, DV_eletric, Towers, MPG, LPS, Pressure_switch, Oil_level, Caudal_impulse) (F=15)
- **Labels:** ⚠️ **No label column.** Labels are built from documented failure windows:
  | Event | Start | End |
  |-------|-------|-----|
  | Air Leak 1 | 2020-04-18 00:00 | 2020-04-18 23:59 |
  | Air Leak 2 | 2020-05-29 23:30 | 2020-05-30 06:00 |
  | Air Leak 3 | 2020-06-05 10:00 | 2020-06-07 14:30 |
  | Air Leak 4 | 2020-07-15 14:30 | 2020-07-15 19:00 |
- **Resampling:** Empirical median interval is 10 seconds (~0.1 Hz). Resampled to 1-minute median (analog) / max (digital), splitting at acquisition gaps over 5 minutes before filling within each block.
- **Window:** L=120, stride=60 (after resampling)
- **Split:** Failure events 1, 2, and 3–4 go to train, validation, and test. Normal-only acquisition blocks are divided chronologically 70/15/15, so every partition has normal and failure observations.
- **⚠️ High-variance results expected:** Only 4 failure events total.
- **Known issue:** The UCI report numbering has a probable typo; one maintenance date precedes a failure window.

---

## Workflow

```
Page 1: Overview        → Understand the pipeline
Page 2: Upload & Inspect → Load your CSV, select columns
Page 3: Preprocessing   → Normalize, split, create windows
Page 4: Teacher Model   → Run CLIP inference (optional)
Page 5: Student Training → Train GRU with/without KD
Page 6: Evaluation      → Compare models, select threshold
Page 7: Inference       → Run on new data, download results
```

### Local website and CSV workflow

This project uses the current Streamlit 1.66 API line. In PowerShell, from the project folder:

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py --server.address=127.0.0.1 --server.port=8501
```

Open `http://127.0.0.1:8501`. The first CSV is the labeled training/reference data: upload it on Page 2, select its time, sensor, and label columns, preprocess it, score the selected windows with the frozen CLIP teacher, train the student, and choose a threshold on validation data. Training saves a checkpoint containing the student, scaler, feature order, imputation settings, and thresholds. On a later app session, load that checkpoint on Page 5 without re-uploading the training data, then upload the target CSV on Page 7. The target CSV must contain the checkpoint's sensor feature columns; extra time, label, or metadata columns are preserved and ignored by the model.

The prediction CSV retains the supplied columns and adds `timestamp`, `timestep_index`, `anomaly_score`, `is_anomaly`, `anomaly_type`, `anomaly_event_id`, event start/end timestamps, and event duration. A one-timestep contiguous detection is labeled `point_anomaly`; a run of two or more adjacent detected timesteps is labeled `group_anomaly`. These describe the shape of model detections and do not by themselves establish the physical cause of an event. If the target CSV contains the training label column, the app also reports point-level metrics for that file.

CLIP is an already-pretrained image/text model and remains frozen. The app loads one shared CLIP instance per server process, scores training and validation window samples, caches soft probabilities for matching data and settings, and passes those probabilities to the student as the KD target. The student is the model used to score newly uploaded CSV files.

---

## Architecture

```
CSV Input
    │
    ▼
┌─────────────────────┐
│  Data Preprocessing  │  sort → fill → normalize → window
└─────────────────────┘
          │
    ┌─────┴──────┐
    │            │
    ▼            ▼
┌────────┐  ┌──────────────────────────────────┐
│ Teacher │  │          Student (GRU)            │
│ (CLIP)  │  │                                  │
│ frozen  │  │  Input: (B, L, F)                │
│         │  │  GRU → window logits (B, 2)      │
│ Image → │  │  GRU → step logits (B, L)        │
│ p_T     │  └──────────────────────────────────┘
└────────┘              │
    │                   │
    └──────────┬─────────┘
               ▼
    ┌──────────────────────┐
    │  Combined Loss       │
    │  L_loc + β·L_win     │
    │       + λ·L_KD       │
    └──────────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │  Threshold Selection │  (on val set only)
    └──────────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │  Anomaly Intervals   │  group consecutive steps
    └──────────────────────┘
```

### Knowledge Distillation Loss

```
L_KD = T² × KL( p_T(T) ‖ p_S(T) )

where:
  p_T(T) = softmax(log(CLIP probabilities) / T)
  p_S(T) = softmax(student_logits / T)
  T       = distillation temperature (configurable)
```

At the default `T=2`, a teacher distribution such as `[0.9, 0.1]` becomes approximately `[0.75, 0.25]`; this makes the target less certain and exposes the relative score of the other class. Multiplying the KL term by `T²` (4 at `T=2`) compensates for the smaller gradients caused by softer distributions. This temperature applies to the KD term only.

### Total Loss

```
L_total = L_localization + β × L_window + λ × L_KD
```

---

## Project Structure

```
IE643 primary assessment/
├── app.py                      # Streamlit entry point
├── config.yaml                 # All hyperparameters
├── requirements.txt
├── README.md
├── src/
│   ├── data/
│   │   ├── loader.py           # CSV loading, load_dataset() interface
│   │   ├── preprocessing.py    # Normalization, windowing, splits
│   │   ├── synthetic_labels.py # Anomaly injection for unlabelled data
│   │   └── datasets/
│   │       ├── creditcard.py
│   │       ├── daphnet.py
│   │       ├── airquality.py
│   │       ├── oil3w.py
│   │       └── metropt.py
│   ├── teacher/
│   │   ├── base.py             # Abstract teacher interface
│   │   ├── image_gen.py        # Window → image converter
│   │   └── clip_teacher.py     # CLIP implementation
│   ├── student/
│   │   ├── model.py            # GRU student model
│   │   └── losses.py           # KD + localization + classification losses
│   ├── training/
│   │   ├── trainer.py          # Training loop + early stopping
│   │   └── checkpoint.py       # Checkpoint save/load
│   ├── evaluation/
│   │   ├── metrics.py          # Precision, recall, F1, AUPRC, ROC-AUC
│   │   ├── threshold.py        # Threshold selection
│   │   └── localization.py     # Interval grouping, overlap metrics
│   ├── visualization/
│   │   ├── plots.py            # Plotly visualizations
│   │   └── report.py           # CSV/ZIP download utilities
│   └── ui/
│       ├── page1_overview.py   # Overview & pipeline
│       ├── page2_upload.py     # Data upload & inspection
│       ├── page3_preprocessing.py
│       ├── page4_teacher.py
│       ├── page5_training.py
│       ├── page6_evaluation.py
│       └── page7_inference.py
├── scripts/
│   ├── generate_synthetic.py   # Generate test data
│   ├── download_daphnet.py
│   ├── download_airquality.py
│   ├── download_3w.py
│   └── download_metropt.py
├── tests/
│   ├── fixtures/               # Tiny CSV files for unit tests
│   └── test_*.py
├── data/                       # Place downloaded datasets here
├── checkpoints/                # Saved model checkpoints
└── cache/                      # Teacher inference cache
```

---

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Run a specific test file
python -m pytest tests/test_preprocessing.py -v
```

---

## Dataset Summary Command

```bash
# After loading a dataset, print per-split statistics:
python -c "
from src.data.loader import load_dataset, summary_command
import yaml
config = yaml.safe_load(open('config.yaml'))
# Replace 'creditcard' with your dataset name
series = load_dataset('creditcard', config)
print(summary_command('creditcard', series, {}))
"
```

---

## Important Limitations

1. **CLIP teacher**: Not trained for time-series. Scores are candidate signals, not calibrated probabilities. Always check teacher evaluation results on the validation set before trusting KD.

2. **Air Quality labels**: All anomalies are synthetically injected. Metrics on this dataset reflect injected anomaly detection performance, not real sensor failure detection.

3. **MetroPT results**: Only 4 failure events → high variance. Do not draw strong conclusions from a single run.

4. **Credit Card**: Temporal structure is weak (different cardholders per row). Window anomaly rate will vary significantly by window assignment.

5. **Model transfer**: A trained student model is tied to the feature set it was trained on. Do not apply it to CSVs with different columns without retraining.

6. **Labels are required for training this objective**: Unlabeled data can be scored after training, but the current GRU loss needs window and timestep labels. Air Quality uses injected labels; those scores are only a synthetic anomaly-recovery experiment.

7. **Final benchmark results are still needed**: The five loaders, split/window paths, frozen CLIP scoring, and one-step GRU/KD updates have been smoke-checked on local data. Full training runs, saved logs/plots, and held-out test metrics still need to be generated before making performance claims. Credit Card transaction ordering and the small number of MetroPT failure events need explicit discussion.

---

## Configuration

All settings are in `config.yaml`. Key parameters:

| Section | Parameter | Default | Description |
|---------|-----------|---------|-------------|
| `preprocessing` | `scaler` | `standard` | StandardScaler or MinMaxScaler |
| `preprocessing` | `missing_strategy` | `interpolate` | How to fill NaN |
| `windowing` | `window_length` | 128 | Time steps per window |
| `windowing` | `stride` | 32 | Stride between windows |
| `teacher` | `model_name` | `openai/clip-vit-base-patch32` | HuggingFace model |
| `distillation` | `temperature` | 2.0 | KD temperature T |
| `distillation` | `lambda_kd` | 0.5 | Weight of KD loss |
| `training` | `epochs` | 30 | Max training epochs |
| `training` | `batch_size` | 32 | Batch size |

---

## Citation / Acknowledgements

- CLIP: Radford et al., "Learning Transferable Visual Models From Natural Language Supervision," ICML 2021
- Daphnet: Bachlin et al., IEEE Trans. Inf. Technol. Biomed. 2010
- 3W: Vargas et al., J. Petrol. Sci. Eng. 2019
- MetroPT: Veloso et al., Data in Brief, 2022

---

*This project was developed for IE643 (Deep Learning). All scientific claims about teacher-student performance require proper experimental validation on held-out test sets.*
