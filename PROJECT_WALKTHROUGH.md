# IE643 Project Walkthrough

## Keep the two projects separate

The attached MRI paper and the course project solve different tasks. The paper is a useful example of anomaly localization by learning normal patterns and finding reconstruction errors. Your allotted project is about multivariate time series, a vision-language teacher, and distillation into a small sequence model. The MRI paper is not the VLM used by this codebase.

## Part 1: What the MRI paper does

The paper is **MCDDPM: Multichannel Conditional Denoising Diffusion Model for Unsupervised Anomaly Detection in Brain MRI** ([arXiv paper](https://arxiv.org/abs/2409.19623)). It treats an anomaly as a brain region that differs from anatomy learned from healthy MRI scans. Its test cases include glioma/tumor regions (including edema and necrotic components) and multiple-sclerosis lesions. It processes 2-D slices extracted from 3-D MRI scans.

### Detection and localization

1. Train only on healthy IXI T2-weighted MRI. The model is not taught tumor masks during training.
2. Use a conditional diffusion model to reconstruct a healthy-looking version of an input MRI slice. The forward diffusion process adds noise at a chosen time step; a U-Net learns the reverse denoising/reconstruction process.
3. MCDDPM adds a multichannel bridge that maps the fully noised image into several latent feature channels, then reconstructs from those channels. Cross-attention connects noisy-image features with clean-image context so the model can use both while reconstructing.
4. Train with two reconstruction objectives: compare the U-Net reconstruction and the bridge reconstruction with the original healthy training image. The paper experiments with L1 and L2 losses.
5. At test time, reconstruct the input as a healthy-looking slice and calculate the pixelwise absolute residual, `R(u,v) = |X(u,v) - X_hat(u,v)|`. Large residuals indicate image locations the healthy model could not explain. Thresholding and image post-processing turn this residual map into a predicted lesion mask.

So “what is the anomaly?” is a region whose appearance is unlike healthy anatomy, and “where is it?” comes from the high-residual pixels. The paper reports better results on its two BraTS tumor sets than on MSLUB; its MSLUB results are close to a random-mask baseline. That is a meaningful limitation: this method should not be described as equally effective across all three datasets.

The paper's reported L2 results are Dice/AUPRC 37.18/41.72 on BraTS 2020, 50.56/57.83 on BraTS 2021, and 5.49/6.67 on MSLUB (percent). The MSLUB Dice is close to the paper's roughly 4% random-mask baseline. These are results reported by the MRI paper; they are not results from the time-series code in this folder.

### How it relates to your assignment

MCDDPM is an unsupervised diffusion/reconstruction method. It does not use a vision-language model, does not distill knowledge, and does not detect multivariate sensor events. Its transferable idea is only the general one: turn a model’s score over time or space into a localized anomaly map.

## Part 2: What the existing code implements

The app is a seven-page Streamlit workflow. It trains a separate student for each dataset and feature schema.

| Stage | What happens |
|---|---|
| 1. Overview | Describes the workflow and model choices. |
| 2. Upload / dataset | Loads a CSV, synthetic example, or one of five dataset adapters; selects time, features, labels, and entity. |
| 3. Preprocess | Keeps recordings separate, splits by time or subject/instance/event, fits imputation/scaling on training data, and builds overlapping windows. |
| 4. Teacher | Plots each selected window as a heatmap or line chart. Frozen CLIP compares it with normal/anomaly text prompts and returns a two-class soft score. |
| 5. Student | Trains a small dual-head GRU: one head classifies the window, one assigns a score to each time step. It supports KD and a supervised-only baseline. |
| 6. Evaluate | Chooses separate window and timestep thresholds on validation data, then reports held-out window, point, and event metrics. |
| 7. Infer | Aggregates overlapping window scores back to a timeline, displays anomaly spans, and exports predictions. |

### Loss and distillation

For a batch, the current student objective is:

```text
L = L_step_BCE + beta * L_window_CE + lambda_KD * T^2 * KL(p_teacher(T) || p_student(T))
```

Here `T=2` is the distillation temperature. It divides the student logits by 2 before softmax, making the student's distribution less peaked. CLIP's probabilities are cached, so the code softens them equivalently with `softmax(log(p_teacher)/T)`. For example, base scores `[0.9, 0.1]` become approximately `[0.75, 0.25]` at `T=2`. The `T^2=4` factor scales the KL term to offset the smaller gradients caused by softening. Temperature affects only the KD term, not the BCE or window cross-entropy terms. If T is too high, the targets approach uniform and lose useful class information.

The GRU learns timestep labels and window labels from the dataset. For KD runs, it also matches CLIP's window-level soft scores on the sampled training windows. The teacher is frozen; only the GRU is optimized. The KD and supervised-only runs must use the same split, model settings, and seed for a fair comparison.

### The CLIP limitation

The code's teacher is generic image-text CLIP, not a model pretrained specifically on time-series. A numerical window is converted to a chart, but CLIP's learned image concepts come from natural image/text pairs. Its softmax output is therefore a candidate heuristic, not a calibrated probability that a sensor anomaly exists. The app now scores validation windows and shows that diagnostic, but the teacher may still be unhelpful or misleading. Keep the supervised-only GRU as a necessary ablation.

Recent task-specific work is available: **VLM4TS** uses a lightweight pretrained vision encoder to screen time-series plots and a VLM to review candidates using wider temporal context ([AAAI 2026 paper](https://ojs.aaai.org/index.php/AAAI/article/view/39319), [official implementation](https://github.com/ZLHe0/VLM4TS)). The published repository's second stage requires an OpenAI API key and is designed for zero-shot interval prediction, so it is a research candidate to compare or adapt—not a drop-in, frozen local teacher for this GRU distillation code. The project's CLIP implementation is a simpler frozen-teacher baseline.

## Requirements check

| Course requirement | Current state | What remains |
|---|---|---|
| At least 4–5 multivariate datasets | The five choices—Credit Card, Daphnet, Air Quality, 3W Oil, and MetroPT—are TA-approved, and adapters exist. | Raw datasets are **not present** in this folder. Download the approved data and check the label/feature mapping for each source. |
| Use pretrained VLM(s) for time series | Frozen CLIP scores rendered time-series charts. | It is a generic VLM with a large domain gap. Explain this clearly and consider a task-specific VLM comparison. |
| Keep VLM frozen | CLIP weights are frozen. | None for the basic requirement. |
| Distill into a smaller CNN/RNN | Dual-head GRU student with window-level KL distillation. | Compare KD against the same GRU trained without KD; don't assume KD helps before running it. |
| Suitable loss | Weighted step BCE + window CE + temperature-scaled KL. | Report the chosen coefficients and justify imbalance handling using validation data. |
| Performance metrics | Window AUPRC/ROC-AUC/F1/precision/recall; point-level localization metrics; event precision/recall/F1; student throughput. | Ask the TAs to approve the primary metric and event-matching rule. Add repeated seeds or uncertainty intervals for a stronger study. |
| Logs, plots, metrics for all selected datasets | The app displays training plots and evaluation plots; it now saves per-run epoch losses and evaluation metrics beside checkpoints. | Train each dataset and preserve the resulting artifacts. No trained checkpoints or measured metrics are present now. |

### Dataset cautions

- **Credit Card Fraud:** 29 numeric features and fraud labels, but row order is transaction order rather than a clean per-card time series. Describe the temporal assumption and avoid claiming natural continuity.
- **Daphnet FoG:** 9 wearable accelerometer channels with freeze labels. Subject-held-out evaluation is a sensible split.
- **Air Quality:** the source has no true anomaly labels. The adapter can inject synthetic anomalies; these measure recovery of injected patterns, not real-world failure detection.
- **3W Oil:** multivariate well telemetry with labeled operating/failure classes. Split instances/wells to avoid near-duplicate leakage.
- **MetroPT-3:** multivariate compressor telemetry with four documented failure episodes. The code holds out event 1 for training, event 2 for validation, events 3–4 for test; results will be high variance because there are few failures.

The recommended primary metric for imbalanced labels is **Average Precision (often reported as AUPRC)**, alongside thresholded precision/recall/F1. For localization, report point-level PR/F1 and event-level precision/recall/F1; show predicted spans over the signal. Use validation data to select thresholds and leave the test set untouched until final evaluation. Accuracy alone is not informative for rare anomalies.

## Important status of the supplied folder

The workspace contains four tiny synthetic CSV examples and source code for five benchmark loaders. It does **not** contain downloaded benchmark data, saved checkpoints, or measured training runs. The generated reports now state that results and runtime comparisons have not been measured; keep that distinction when presenting the project and add held-out outputs only after running the experiments.

The app also requires ground-truth labels to train its current loss. An unlabeled time series is accepted for inference, but cannot train this supervised localization objective without a separately designed unsupervised or pseudo-label objective.

## Run sequence

1. Install `requirements.txt` in a Python environment with PyTorch, Streamlit, scikit-learn, Transformers, and the plotting dependencies.
2. Download the raw files for the datasets selected with the TAs. The five scripts in `scripts/` are dataset-specific; Credit Card data is a manual Kaggle download.
3. Launch `streamlit run app.py`.
4. Load one dataset, inspect its label semantics, select numeric features, and confirm the mapping.
5. Preprocess and inspect split counts, anomaly counts, and split roles. Fix window length if any split has no complete windows or anomalies.
6. Score training windows with the frozen teacher and inspect its validation AUPRC/ROC-AUC. If the teacher is poor, keep that result and compare against the supervised-only baseline.
7. Train both the supervised-only GRU and the KD GRU on the same dataset/configuration. Repeat across all chosen datasets.
8. Tune thresholds only on validation data, evaluate once on test, and save the timeline plots, metric CSVs, training curves, and checkpoint for each run.

The dataset choices are already TA-approved. The code remains an implementation scaffold until the approved files are acquired, the metric protocol is confirmed, and the experiments are run.
