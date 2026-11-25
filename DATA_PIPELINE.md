# What the anomaly detection pipeline does

The application loads each source as one or more separate time-series sequences. Each sequence contains a float32 sensor matrix `(time steps, features)`, point labels, timestamps, and an identity such as subject, recording, well instance, or acquisition block. The app does not make windows across sequence boundaries.

## Dataset-specific processing

| Dataset | Labels and preparation | Split and windows |
|---|---|---|
| Credit Card Fraud | Sort by elapsed `Time`; use V1–V28 and `Amount` (29 features); `Class=1` is fraud. | Chronological 60/20/20; 64 steps per window, stride 32. |
| Daphnet FoG | Read 11-column recordings; drop annotation 0; annotation 1 is normal and 2 is freezing of gait; retain 9 acceleration channels. Rows dropped between valid stretches form sequence boundaries. | Hold out subjects; 256 steps (4 seconds at 64 Hz), stride 64. |
| UCI Air Quality | Combine date/time and sort; convert `-200` to missing; drop features with over 50% missing (`NMHC(GT)` in this copy), leaving 12 features. Impute after the chronological split and fit scaling on training data only. The default label strategy injects labeled spikes, shifts, drifts, and noise bursts independently into train, validation, and test, using feature scales from clean training data. | Chronological 70/15/15; 48 steps, stride 12. Scores measure recovery of injected patterns, since this dataset has no native anomaly labels. |
| Petrobras 3W | Load all 1,984 CSV instances; use eight pressure, temperature, and flow features. Class 0 is normal; all nonzero classes, including transient codes 101–108, are anomalies. Treat NaN, +/-inf, and values outside float32 range as missing, then fill within each file using forward fill, backward fill, and zero for any remaining values. | Hold out well instances; 120 steps, stride 60. The default training subset is at most 20,000 windows, allocated across instances and labels to control memory. Training-time validation uses at most 5,000 windows; full validation and test windows are retained for threshold selection and final metrics. |
| MetroPT-3 | Use 7 analog and 8 digital channels. The observed raw interval is about 10 seconds. Break the timeline at acquisition gaps over five minutes, then resample each block to one minute (analog median, digital maximum) and fill only within that block. Annotate the four published failure intervals. | Event 1 goes to train, event 2 to validation, events 3–4 to test. Normal-only acquisition blocks are divided chronologically 70/15/15, so every split has normal and failure data. Use 120 steps, stride 60. |

For all datasets, imputation references and the feature scaler are fit from training data. Sliding windows receive a positive window label when any step in the window is anomalous by default. The chosen length and stride come from `config.yaml` and can be changed in the preprocessing page. The 3W training-window cap is visible there; set it to zero only if the machine has enough memory for all windows.

## Teacher, student, and temperature

1. The frozen `openai/clip-vit-base-patch32` model receives a 224×224 heatmap rendered from the scaled sensor window. It compares that image with one normal and one anomaly text prompt. CLIP returns a two-class probability distribution. The default scoring budget is 1,000 training windows and up to 2,000 validation windows; KD and its paired supervised baseline train on the scored training subset. CLIP was pretrained on image/text data, not time-series anomaly labels, so its output is a candidate teacher signal and must be compared with the labeled baseline.
2. The student is a two-layer GRU by default. One head predicts whether a window is anomalous; the other predicts an anomaly score for every step, which is used to localize spans on the plotted timeline.
3. Training combines weighted step-level binary cross-entropy, window-level cross-entropy, and teacher/student KL divergence. At temperature `T=2`, dividing logits by two softens both distributions so the student can learn from less-certain teacher preferences. The KL term is multiplied by `T²` to preserve its gradient scale. The supervised labels still provide the localization target.
4. Validation data selects decision thresholds. Final reported window, step, and event metrics use the held-out test partition. The main score set includes AUPRC, ROC-AUC, F1, precision/recall, and event precision/recall/F1.

## What has been checked

The local copies loaded successfully: Credit Card (284,807 rows), Daphnet (1,140,835 valid steps across 35 segments and 10 subjects), Air Quality (9,357 rows), 3W (50,913,215 steps across 1,984 instances), and MetroPT (307,198 one-minute steps across 276 acquisition blocks). A frozen CLIP teacher scored a rendered window from the local Air Quality file. A GRU forward/backward optimization step with the production distillation loss ran on a real training sequence from each dataset. The full 3W run also fit the train-only incremental scaler on all training instances, verified the 20,000-window training cap, and completed a student/KD step. The project regression suite passes (79 tests).

These are integration smoke checks, not benchmark results. Full multi-epoch training, comparison against the supervised-only baseline, and final held-out metrics/plots still need to be generated for the project report. Air Quality results remain synthetic; MetroPT has only four known failures; Credit Card row order does not track individual cardholders.
