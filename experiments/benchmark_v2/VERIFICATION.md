# Verification of the v2 results

## A. Independent recomputation of saved test metrics

For every dataset and model, the seed-0 test timeline scores saved in `test_predictions_seed0.npz` are re-scored with `sklearn.metrics.average_precision_score` and compared with the value in `metrics.csv`.

| Dataset | Model | Prevalence | AUPRC (metrics.csv) | AUPRC (recomputed) | AUPRC of scores×100 | Random scores | Shuffled labels |
|---|---|---|---|---|---|---|---|
| creditcard | clip_zeroshot | 0.0013 | 0.0014 | 0.0014 | 0.0014 | 0.0015 | 0.0013 |
| creditcard | clip_probe | 0.0013 | 0.0061 | 0.0061 | 0.0061 | 0.0016 | 0.0014 |
| creditcard | isolation_forest | 0.0013 | 0.0029 | 0.0029 | 0.0029 | 0.0014 | 0.0016 |
| creditcard | gbdt | 0.0013 | 0.0237 | 0.0237 | 0.0237 | 0.0014 | 0.0014 |
| creditcard | gru_gt | 0.0013 | 0.7946 | 0.7946 | 0.7946 | 0.0013 | 0.0016 |
| creditcard | gru_kd_zeroshot | 0.0013 | 0.7905 | 0.7905 | 0.7905 | 0.0015 | 0.0014 |
| creditcard | gru_kd_probe | 0.0013 | 0.7736 | 0.7736 | 0.7736 | 0.0015 | 0.0014 |
| creditcard | gru_kd_full | 0.0013 | 0.7794 | 0.7794 | 0.7794 | 0.0016 | 0.0014 |
| creditcard | gru_teacher_only | 0.0013 | 0.7622 | 0.7622 | 0.7622 | 0.0016 | 0.0015 |
| creditcard | clip_probe_mv | 0.0013 | 0.0047 | 0.0047 | 0.0047 | 0.0014 | 0.0014 |
| creditcard | clip_coop | 0.0013 | 0.0030 | 0.0030 | 0.0030 | 0.0014 | 0.0012 |
| creditcard | clip_coop_mv | 0.0013 | 0.0027 | 0.0027 | 0.0027 | 0.0014 | 0.0015 |
| creditcard | gru_kd_coop_plain | 0.0013 | 0.7872 | 0.7872 | 0.7872 | 0.0013 | 0.0015 |
| creditcard | gru_kd_coop | 0.0013 | 0.7776 | 0.7776 | 0.7776 | 0.0014 | 0.0014 |
| daphnet | clip_zeroshot | 0.0221 | 0.0591 | 0.0591 | 0.0591 | 0.0223 | 0.0221 |
| daphnet | clip_probe | 0.0221 | 0.4040 | 0.4040 | 0.4040 | 0.0222 | 0.0222 |
| daphnet | isolation_forest | 0.0221 | 0.0301 | 0.0301 | 0.0301 | 0.0222 | 0.0222 |
| daphnet | gbdt | 0.0221 | 0.2111 | 0.2111 | 0.2111 | 0.0219 | 0.0223 |
| daphnet | gru_gt | 0.0221 | 0.5722 | 0.5722 | 0.5722 | 0.0222 | 0.0223 |
| daphnet | gru_kd_zeroshot | 0.0221 | 0.4903 | 0.4903 | 0.4903 | 0.0222 | 0.0221 |
| daphnet | gru_kd_probe | 0.0221 | 0.5876 | 0.5876 | 0.5876 | 0.0222 | 0.0222 |
| daphnet | gru_kd_full | 0.0221 | 0.5853 | 0.5853 | 0.5853 | 0.0222 | 0.0222 |
| daphnet | gru_teacher_only | 0.0221 | 0.5222 | 0.5222 | 0.5222 | 0.0224 | 0.0221 |
| daphnet | clip_probe_mv | 0.0221 | 0.4047 | 0.4047 | 0.4047 | 0.0222 | 0.0219 |
| daphnet | clip_coop | 0.0221 | 0.4711 | 0.4711 | 0.4711 | 0.0223 | 0.0224 |
| daphnet | clip_coop_mv | 0.0221 | 0.4523 | 0.4523 | 0.4523 | 0.0223 | 0.0224 |
| daphnet | gru_kd_coop_plain | 0.0221 | 0.5142 | 0.5142 | 0.5142 | 0.0223 | 0.0221 |
| daphnet | gru_kd_coop | 0.0221 | 0.5124 | 0.5124 | 0.5124 | 0.0221 | 0.0220 |
| metropt | clip_zeroshot | 0.0637 | 0.0386 | 0.0386 | 0.0386 | 0.0639 | 0.0636 |
| metropt | clip_probe | 0.0637 | 0.9419 | 0.9419 | 0.9419 | 0.0627 | 0.0641 |
| metropt | isolation_forest | 0.0637 | 0.6076 | 0.6076 | 0.6076 | 0.0638 | 0.0635 |
| metropt | gbdt | 0.0637 | 0.6915 | 0.6915 | 0.6915 | 0.0642 | 0.0637 |
| metropt | gru_gt | 0.0637 | 0.7520 | 0.7520 | 0.7520 | 0.0638 | 0.0639 |
| metropt | gru_kd_zeroshot | 0.0637 | 0.8099 | 0.8099 | 0.8099 | 0.0645 | 0.0634 |
| metropt | gru_kd_probe | 0.0637 | 0.8610 | 0.8610 | 0.8610 | 0.0649 | 0.0640 |
| metropt | gru_kd_full | 0.0637 | 0.8598 | 0.8598 | 0.8598 | 0.0634 | 0.0636 |
| metropt | gru_teacher_only | 0.0637 | 0.8441 | 0.8441 | 0.8441 | 0.0635 | 0.0629 |
| metropt | clip_probe_mv | 0.0637 | 0.5896 | 0.5896 | 0.5896 | 0.0643 | 0.0637 |
| metropt | clip_coop | 0.0637 | 0.9905 | 0.9905 | 0.9905 | 0.0636 | 0.0638 |
| metropt | clip_coop_mv | 0.0637 | 0.9994 | 0.9994 | 0.9994 | 0.0642 | 0.0642 |
| metropt | gru_kd_coop_plain | 0.0637 | 0.8708 | 0.8708 | 0.8708 | 0.0639 | 0.0640 |
| metropt | gru_kd_coop | 0.0637 | 0.8835 | 0.8835 | 0.8835 | 0.0630 | 0.0638 |
| oil3w | clip_zeroshot | 0.6037 | 0.7692 | 0.7692 | 0.7692 | 0.6050 | 0.6037 |
| oil3w | clip_probe | 0.6037 | 0.9821 | 0.9821 | 0.9821 | 0.6038 | 0.6042 |
| oil3w | isolation_forest | 0.6037 | 0.7694 | 0.7694 | 0.7694 | 0.6017 | 0.6038 |
| oil3w | gbdt | 0.6037 | 0.9925 | 0.9925 | 0.9925 | 0.6041 | 0.6043 |
| oil3w | gru_gt | 0.6037 | 0.9800 | 0.9800 | 0.9800 | 0.6038 | 0.6041 |
| oil3w | gru_kd_zeroshot | 0.6037 | 0.9779 | 0.9779 | 0.9779 | 0.6041 | 0.6043 |
| oil3w | gru_kd_probe | 0.6037 | 0.9800 | 0.9800 | 0.9800 | 0.6036 | 0.6035 |
| oil3w | gru_kd_full | 0.6037 | 0.9727 | 0.9727 | 0.9727 | 0.6038 | 0.6031 |
| oil3w | gru_teacher_only | 0.6037 | 0.9732 | 0.9732 | 0.9732 | 0.6023 | 0.6038 |
| oil3w | clip_probe_mv | 0.6037 | 0.9813 | 0.9813 | 0.9813 | 0.6036 | 0.6030 |
| oil3w | clip_coop | 0.6037 | 0.9640 | 0.9640 | 0.9640 | 0.6037 | 0.6031 |
| oil3w | clip_coop_mv | 0.6037 | 0.9597 | 0.9597 | 0.9597 | 0.6041 | 0.6040 |
| oil3w | gru_kd_coop_plain | 0.6037 | 0.9814 | 0.9814 | 0.9814 | 0.6030 | 0.6029 |
| oil3w | gru_kd_coop | 0.6037 | 0.9796 | 0.9796 | 0.9796 | 0.6038 | 0.6052 |

Largest difference between saved and recomputed AUPRC: 0.00e+00. AUPRC depends only on the ranking of scores, so multiplying scores by 100 leaves it unchanged, and it can never exceed 1. A random detector scores at the prevalence, and shuffling the labels drops every model to the prevalence. So the reported values come from real agreement between scores and labels.

## B. Why the Daphnet score rose: step-by-step attribution on the identical test split

Test split check: v1 test subjects ['S04', 'S07'], 5213 anomalous steps, 176 anomalous windows; v2 test subjects ['S04', 'S07'], 5213 anomalous steps, 176 anomalous windows. Window prevalence 0.0479, step prevalence 0.0221.

| Stage | Model | Test window AUPRC | Test point AUPRC | Note |
|---|---|---|---|---|
| T1 | v1 teacher: zero-shot CLIP, global-range heatmap, 2 prompts | 0.029 | – | v1 artefact |
| T2 | v2 teacher: zero-shot CLIP, clipped heatmap, 8 prompts | 0.123 | 0.059 | rendering fix |
| T3 | v2 teacher: linear probe on frozen CLIP embeddings | 0.542 | 0.404 | adaptation without changing CLIP |
| T4 | v2.1 teacher: learned prompts (CoOp), heatmap | 0.531 | 0.471 | prompt learning, CLIP frozen |
| T5 | v2.1 teacher: learned prompts, heatmap + spectrogram | 0.401 | 0.452 | final teacher |
| S0 | v1 UI checkpoint (4 Nov; 2-layer GRU, 50 teacher windows) | 0.058 | 0.056 | reproduces the v1 report |
| S1 | v1 controlled study: GRU, labels only, raw scores | 0.280 | 0.224 | same model, v1 scoring |
| S2 | v1 controlled study: GRU, labels only + v2 localization | 0.280 | 0.289 | alpha=1.0, k=64 |
| S1 | v1 controlled study: GRU + zero-shot KD, raw scores | 0.394 | 0.307 | same model, v1 scoring |
| S2 | v1 controlled study: GRU + zero-shot KD + v2 localization | 0.394 | 0.377 | alpha=1.0, k=64 |
| S3 | v2 GRU, labels only: mean of 3 seeds | 0.627 | 0.519 | conv+BiGRU, AUPRC selection |
| S4 | v2 GRU, labels only: deployed checkpoint (seed 1), raw scores | 0.562 | 0.386 | no post-processing |
| S3 | v2 GRU + KD (probe + features): mean of 3 seeds | 0.640 | 0.523 | conv+BiGRU, AUPRC selection |
| S4 | v2 GRU + KD (probe + features): deployed checkpoint (seed 1), raw scores | 0.519 | 0.459 | no post-processing |
| S3 | v2 GRU + KD (learned prompts, 2 views, confidence-weighted): mean of 3 seeds | 0.592 | 0.477 | conv+BiGRU, AUPRC selection |
| S4 | v2 GRU + KD (learned prompts, 2 views, confidence-weighted): deployed checkpoint (seed 1), raw scores | 0.514 | 0.449 | no post-processing |

Random detector on this split: window AUPRC 0.048, point AUPRC 0.022.

## C. Leakage and protocol checks

- **creditcard**: chronological 60%/20%/20%; group overlap between partitions: none; test window prevalence 0.074.
- **daphnet**: subject-disjoint stratified group split (seed 42); group overlap between partitions: none; test window prevalence 0.048.
- **metropt**: dataset-provided roles (MetroPT event holdout); group overlap between partitions: none; test window prevalence 0.062.
- **oil3w**: instance-disjoint stratified group split (seed 42); group overlap between partitions: none; test window prevalence 0.620.
- Scaler and imputation statistics are fitted on training sequences only (`src/experiments/data.py`).
- Probe regularisation is chosen by cross-validation on training windows; its training-window soft labels are out-of-fold.
- Early stopping, window/step thresholds, gating, smoothing and minimum event length use validation data only. Test labels are read only by the final metric computation.
- Air Quality is excluded (synthetic labels); its audited runs are in `experiments/excluded_airquality/`.
- Reproducibility: re-running a model with the same seed and thread count reproduces its metrics exactly (MetroPT labels-only GRU, seed 0: 0.798 window / 0.752 point AUPRC in both runs). Changing the CPU thread count changes floating-point summation order, so other seeds can differ between runs; all reported numbers come from the final runs listed in each dataset's metrics.csv.
- Thresholds: F1-optimal on validation, except MetroPT (1 validation event), where the threshold is the 99.9th percentile of normal validation scores (0.1% false-alarm rate).
