# v2 benchmark results

All numbers are on the held-out **test** split. Thresholds, smoothing, gating and minimum event length were chosen on validation data only. Students: mean ± std over seeds 0, 1, 2 with identical splits and identical initial weights per seed. AUPRC of a random detector equals the prevalence.

## Credit Card

Split: chronological 60%/20%/20%. Test windows: 1780 (0.074 anomalous); test timesteps: 56961 (0.0013 anomalous).

Teacher soft-label quality on the training windows it labels (out-of-fold window AUPRC; prevalence 0.098): zeroshot 0.118, probe 0.462, probe_mv 0.462, coop 0.332, coop_mv 0.321.

| Model | Window AUPRC | Window F1 | Point AUPRC | Point F1 | Event F1 |
|---|---|---|---|---|---|
| CLIP zero-shot (frozen) | 0.079 | 0.144 | 0.001 | 0.001 | 0.018 |
| CLIP linear probe (frozen backbone) | 0.217 | 0.230 | 0.006 | 0.000 | 0.000 |
| CLIP linear probe, heatmap + spectrogram | 0.191 | 0.282 | 0.005 | 0.000 | 0.000 |
| CLIP learned prompts (CoOp) | 0.137 | 0.228 | 0.003 | 0.000 | 0.000 |
| CLIP learned prompts, heatmap + spectrogram | 0.124 | 0.202 | 0.003 | 0.000 | 0.000 |
| Isolation Forest | 0.128 | 0.206 | 0.003 | 0.006 | 0.265 |
| Gradient-boosted trees | 0.767 | 0.770 | 0.024 | 0.066 | 0.622 |
| GRU, ground truth only | 0.823 ± 0.005 | 0.825 ± 0.013 | 0.787 ± 0.010 | 0.802 ± 0.014 | 0.799 ± 0.014 |
| GRU + KD (zero-shot CLIP) | 0.817 ± 0.001 | 0.826 ± 0.008 | 0.786 ± 0.004 | 0.792 ± 0.016 | 0.789 ± 0.016 |
| GRU + KD (probe soft labels) | 0.803 ± 0.008 | 0.813 ± 0.035 | 0.778 ± 0.008 | 0.805 ± 0.018 | 0.802 ± 0.019 |
| GRU + KD (soft labels + CLIP features) | 0.808 ± 0.010 | 0.807 ± 0.039 | 0.783 ± 0.006 | 0.808 ± 0.017 | 0.805 ± 0.017 |
| GRU, window head from teacher only (beta=0) | 0.770 ± 0.034 | 0.735 ± 0.037 | 0.760 ± 0.014 | 0.820 ± 0.006 | 0.820 ± 0.007 |
| GRU + KD (learned prompts, 2 views) | 0.804 ± 0.005 | 0.816 ± 0.023 | 0.779 ± 0.008 | 0.799 ± 0.009 | 0.800 ± 0.006 |
| GRU + KD (learned prompts, 2 views, confidence-weighted) | 0.801 ± 0.011 | 0.826 ± 0.011 | 0.775 ± 0.004 | 0.791 ± 0.008 | 0.792 ± 0.006 |

## Daphnet FoG

Split: subject-disjoint stratified group split (seed 42). Test windows: 3672 (0.048 anomalous); test timesteps: 235524 (0.0221 anomalous).

Teacher soft-label quality on the training windows it labels (out-of-fold window AUPRC; prevalence 0.171): zeroshot 0.352, probe 0.760, probe_mv 0.761, coop 0.707, coop_mv 0.645.

| Model | Window AUPRC | Window F1 | Point AUPRC | Point F1 | Event F1 |
|---|---|---|---|---|---|
| CLIP zero-shot (frozen) | 0.123 | 0.161 | 0.059 | 0.079 | 0.222 |
| CLIP linear probe (frozen backbone) | 0.542 | 0.300 | 0.404 | 0.280 | 0.351 |
| CLIP linear probe, heatmap + spectrogram | 0.573 | 0.282 | 0.405 | 0.372 | 0.510 |
| CLIP learned prompts (CoOp) | 0.531 | 0.355 | 0.471 | 0.293 | 0.421 |
| CLIP learned prompts, heatmap + spectrogram | 0.401 | 0.293 | 0.452 | 0.221 | 0.373 |
| Isolation Forest | 0.057 | 0.099 | 0.030 | 0.062 | 0.194 |
| Gradient-boosted trees | 0.350 | 0.183 | 0.211 | 0.137 | 0.246 |
| GRU, ground truth only | 0.627 ± 0.058 | 0.289 ± 0.078 | 0.519 ± 0.062 | 0.255 ± 0.102 | 0.345 ± 0.105 |
| GRU + KD (zero-shot CLIP) | 0.609 ± 0.057 | 0.253 ± 0.031 | 0.464 ± 0.031 | 0.174 ± 0.051 | 0.240 ± 0.055 |
| GRU + KD (probe soft labels) | 0.602 ± 0.177 | 0.344 ± 0.125 | 0.504 ± 0.092 | 0.238 ± 0.103 | 0.317 ± 0.035 |
| GRU + KD (soft labels + CLIP features) | 0.640 ± 0.110 | 0.330 ± 0.125 | 0.523 ± 0.062 | 0.241 ± 0.101 | 0.319 ± 0.071 |
| GRU, window head from teacher only (beta=0) | 0.601 ± 0.088 | 0.270 ± 0.085 | 0.506 ± 0.034 | 0.205 ± 0.099 | 0.290 ± 0.082 |
| GRU + KD (learned prompts, 2 views) | 0.587 ± 0.089 | 0.248 ± 0.046 | 0.481 ± 0.029 | 0.159 ± 0.043 | 0.258 ± 0.019 |
| GRU + KD (learned prompts, 2 views, confidence-weighted) | 0.592 ± 0.068 | 0.238 ± 0.022 | 0.477 ± 0.031 | 0.147 ± 0.028 | 0.252 ± 0.044 |

## 3W Oil

Split: instance-disjoint stratified group split (seed 42). Test windows: 3585 (0.620 anomalous); test timesteps: 112079 (0.6037 anomalous).

Teacher soft-label quality on the training windows it labels (out-of-fold window AUPRC; prevalence 0.651): zeroshot 0.817, probe 0.972, probe_mv 0.977, coop 0.950, coop_mv 0.940.

| Model | Window AUPRC | Window F1 | Point AUPRC | Point F1 | Event F1 |
|---|---|---|---|---|---|
| CLIP zero-shot (frozen) | 0.771 | 0.722 | 0.769 | 0.717 | 0.745 |
| CLIP linear probe (frozen backbone) | 0.992 | 0.972 | 0.982 | 0.960 | 0.971 |
| CLIP linear probe, heatmap + spectrogram | 0.991 | 0.974 | 0.981 | 0.964 | 0.943 |
| CLIP learned prompts (CoOp) | 0.974 | 0.963 | 0.964 | 0.951 | 0.956 |
| CLIP learned prompts, heatmap + spectrogram | 0.971 | 0.931 | 0.960 | 0.923 | 0.912 |
| Isolation Forest | 0.787 | 0.765 | 0.769 | 0.753 | 0.736 |
| Gradient-boosted trees | 0.998 | 0.995 | 0.992 | 0.990 | 1.000 |
| GRU, ground truth only | 0.985 ± 0.003 | 0.961 ± 0.006 | 0.978 ± 0.003 | 0.948 ± 0.008 | 0.925 ± 0.039 |
| GRU + KD (zero-shot CLIP) | 0.982 ± 0.002 | 0.959 ± 0.002 | 0.978 ± 0.001 | 0.949 ± 0.001 | 0.949 ± 0.008 |
| GRU + KD (probe soft labels) | 0.985 ± 0.006 | 0.941 ± 0.010 | 0.981 ± 0.007 | 0.932 ± 0.025 | 0.920 ± 0.031 |
| GRU + KD (soft labels + CLIP features) | 0.980 ± 0.004 | 0.938 ± 0.019 | 0.976 ± 0.004 | 0.934 ± 0.016 | 0.875 ± 0.109 |
| GRU, window head from teacher only (beta=0) | 0.973 ± 0.003 | 0.941 ± 0.011 | 0.975 ± 0.003 | 0.940 ± 0.013 | 0.924 ± 0.030 |
| GRU + KD (learned prompts, 2 views) | 0.978 ± 0.001 | 0.953 ± 0.016 | 0.980 ± 0.002 | 0.953 ± 0.001 | 0.943 ± 0.009 |
| GRU + KD (learned prompts, 2 views, confidence-weighted) | 0.978 ± 0.004 | 0.946 ± 0.014 | 0.978 ± 0.002 | 0.943 ± 0.010 | 0.943 ± 0.009 |

## MetroPT-3

Split: dataset-provided roles (MetroPT event holdout). Test windows: 1685 (0.062 anomalous); test timesteps: 54208 (0.0631 anomalous).

Teacher soft-label quality on the training windows it labels (out-of-fold window AUPRC; prevalence 0.007): zeroshot 0.004, probe 0.418, probe_mv 0.994, coop 0.219, coop_mv 0.842.

| Model | Window AUPRC | Window F1 | Point AUPRC | Point F1 | Event F1 |
|---|---|---|---|---|---|
| CLIP zero-shot (frozen) | 0.037 | 0.000 | 0.039 | 0.000 | 0.000 |
| CLIP linear probe (frozen backbone) | 0.931 | 0.826 | 0.942 | 0.838 | 0.889 |
| CLIP linear probe, heatmap + spectrogram | 0.345 | 0.250 | 0.590 | 0.460 | 0.774 |
| CLIP learned prompts (CoOp) | 0.989 | 0.839 | 0.991 | 0.843 | 0.889 |
| CLIP learned prompts, heatmap + spectrogram | 0.998 | 0.805 | 0.999 | 0.470 | 0.857 |
| Isolation Forest | 0.575 | 0.019 | 0.608 | 0.017 | 0.333 |
| Gradient-boosted trees | 0.702 | 0.816 | 0.692 | 0.826 | 0.800 |
| GRU, ground truth only | 0.751 ± 0.137 | 0.439 ± 0.396 | 0.524 ± 0.251 | 0.260 ± 0.451 | 0.250 ± 0.433 |
| GRU + KD (zero-shot CLIP) | 0.424 ± 0.306 | 0.457 ± 0.396 | 0.824 ± 0.037 | 0.665 ± 0.131 | 0.641 ± 0.209 |
| GRU + KD (probe soft labels) | 0.860 ± 0.030 | 0.782 ± 0.027 | 0.858 ± 0.041 | 0.797 ± 0.006 | 0.758 ± 0.014 |
| GRU + KD (soft labels + CLIP features) | 0.865 ± 0.021 | 0.742 ± 0.040 | 0.870 ± 0.010 | 0.796 ± 0.010 | 0.750 |
| GRU, window head from teacher only (beta=0) | 0.842 ± 0.038 | 0.771 ± 0.026 | 0.828 ± 0.023 | 0.793 ± 0.016 | 0.767 ± 0.029 |
| GRU + KD (learned prompts, 2 views) | 0.893 ± 0.018 | 0.805 | 0.890 ± 0.017 | 0.804 ± 0.003 | 0.796 ± 0.080 |
| GRU + KD (learned prompts, 2 views, confidence-weighted) | 0.876 ± 0.015 | 0.794 ± 0.019 | 0.889 ± 0.006 | 0.796 ± 0.011 | 0.722 ± 0.048 |

## Paired comparison with the ground-truth-only GRU

Mean change over seeds (same split, same initial weights) and the number of seeds where the variant beats the ground-truth-only GRU.

| Dataset | Level | Variant | Mean Δ | Wins |
|---|---|---|---|---|
| Credit Card | event (event_f1) | GRU + KD (learned prompts, 2 views, confidence-weighted) | -0.007 | 1/3 |
| Credit Card | event (event_f1) | GRU + KD (learned prompts, 2 views) | +0.002 | 1/3 |
| Credit Card | event (event_f1) | GRU + KD (soft labels + CLIP features) | +0.006 | 2/3 |
| Credit Card | event (event_f1) | GRU + KD (probe soft labels) | +0.003 | 2/3 |
| Credit Card | event (event_f1) | GRU + KD (zero-shot CLIP) | -0.010 | 1/3 |
| Credit Card | event (event_f1) | GRU, window head from teacher only (beta=0) | +0.021 | 2/3 |
| Credit Card | point (auprc) | GRU + KD (learned prompts, 2 views, confidence-weighted) | -0.012 | 1/3 |
| Credit Card | point (auprc) | GRU + KD (learned prompts, 2 views) | -0.008 | 1/3 |
| Credit Card | point (auprc) | GRU + KD (soft labels + CLIP features) | -0.004 | 1/3 |
| Credit Card | point (auprc) | GRU + KD (probe soft labels) | -0.009 | 1/3 |
| Credit Card | point (auprc) | GRU + KD (zero-shot CLIP) | -0.000 | 1/3 |
| Credit Card | point (auprc) | GRU, window head from teacher only (beta=0) | -0.026 | 0/3 |
| Credit Card | window (auprc) | GRU + KD (learned prompts, 2 views, confidence-weighted) | -0.022 | 0/3 |
| Credit Card | window (auprc) | GRU + KD (learned prompts, 2 views) | -0.019 | 0/3 |
| Credit Card | window (auprc) | GRU + KD (soft labels + CLIP features) | -0.016 | 0/3 |
| Credit Card | window (auprc) | GRU + KD (probe soft labels) | -0.020 | 0/3 |
| Credit Card | window (auprc) | GRU + KD (zero-shot CLIP) | -0.007 | 0/3 |
| Credit Card | window (auprc) | GRU, window head from teacher only (beta=0) | -0.053 | 0/3 |
| Daphnet FoG | event (event_f1) | GRU + KD (learned prompts, 2 views, confidence-weighted) | -0.093 | 1/3 |
| Daphnet FoG | event (event_f1) | GRU + KD (learned prompts, 2 views) | -0.087 | 1/3 |
| Daphnet FoG | event (event_f1) | GRU + KD (soft labels + CLIP features) | -0.026 | 2/3 |
| Daphnet FoG | event (event_f1) | GRU + KD (probe soft labels) | -0.029 | 1/3 |
| Daphnet FoG | event (event_f1) | GRU + KD (zero-shot CLIP) | -0.106 | 0/3 |
| Daphnet FoG | event (event_f1) | GRU, window head from teacher only (beta=0) | -0.055 | 1/3 |
| Daphnet FoG | point (auprc) | GRU + KD (learned prompts, 2 views, confidence-weighted) | -0.042 | 1/3 |
| Daphnet FoG | point (auprc) | GRU + KD (learned prompts, 2 views) | -0.038 | 1/3 |
| Daphnet FoG | point (auprc) | GRU + KD (soft labels + CLIP features) | +0.004 | 2/3 |
| Daphnet FoG | point (auprc) | GRU + KD (probe soft labels) | -0.015 | 1/3 |
| Daphnet FoG | point (auprc) | GRU + KD (zero-shot CLIP) | -0.055 | 1/3 |
| Daphnet FoG | point (auprc) | GRU, window head from teacher only (beta=0) | -0.013 | 1/3 |
| Daphnet FoG | window (auprc) | GRU + KD (learned prompts, 2 views, confidence-weighted) | -0.035 | 0/3 |
| Daphnet FoG | window (auprc) | GRU + KD (learned prompts, 2 views) | -0.040 | 0/3 |
| Daphnet FoG | window (auprc) | GRU + KD (soft labels + CLIP features) | +0.013 | 2/3 |
| Daphnet FoG | window (auprc) | GRU + KD (probe soft labels) | -0.025 | 2/3 |
| Daphnet FoG | window (auprc) | GRU + KD (zero-shot CLIP) | -0.018 | 0/3 |
| Daphnet FoG | window (auprc) | GRU, window head from teacher only (beta=0) | -0.026 | 1/3 |
| MetroPT-3 | event (event_f1) | GRU + KD (learned prompts, 2 views, confidence-weighted) | +0.472 | 2/3 |
| MetroPT-3 | event (event_f1) | GRU + KD (learned prompts, 2 views) | +0.546 | 2/3 |
| MetroPT-3 | event (event_f1) | GRU + KD (soft labels + CLIP features) | +0.500 | 2/3 |
| MetroPT-3 | event (event_f1) | GRU + KD (probe soft labels) | +0.508 | 3/3 |
| MetroPT-3 | event (event_f1) | GRU + KD (zero-shot CLIP) | +0.391 | 3/3 |
| MetroPT-3 | event (event_f1) | GRU, window head from teacher only (beta=0) | +0.517 | 2/3 |
| MetroPT-3 | point (auprc) | GRU + KD (learned prompts, 2 views, confidence-weighted) | +0.365 | 3/3 |
| MetroPT-3 | point (auprc) | GRU + KD (learned prompts, 2 views) | +0.366 | 3/3 |
| MetroPT-3 | point (auprc) | GRU + KD (soft labels + CLIP features) | +0.345 | 3/3 |
| MetroPT-3 | point (auprc) | GRU + KD (probe soft labels) | +0.334 | 3/3 |
| MetroPT-3 | point (auprc) | GRU + KD (zero-shot CLIP) | +0.299 | 3/3 |
| MetroPT-3 | point (auprc) | GRU, window head from teacher only (beta=0) | +0.304 | 3/3 |
| MetroPT-3 | window (auprc) | GRU + KD (learned prompts, 2 views, confidence-weighted) | +0.124 | 3/3 |
| MetroPT-3 | window (auprc) | GRU + KD (learned prompts, 2 views) | +0.142 | 3/3 |
| MetroPT-3 | window (auprc) | GRU + KD (soft labels + CLIP features) | +0.114 | 2/3 |
| MetroPT-3 | window (auprc) | GRU + KD (probe soft labels) | +0.109 | 3/3 |
| MetroPT-3 | window (auprc) | GRU + KD (zero-shot CLIP) | -0.327 | 0/3 |
| MetroPT-3 | window (auprc) | GRU, window head from teacher only (beta=0) | +0.091 | 2/3 |
| 3W Oil | event (event_f1) | GRU + KD (learned prompts, 2 views, confidence-weighted) | +0.018 | 1/3 |
| 3W Oil | event (event_f1) | GRU + KD (learned prompts, 2 views) | +0.018 | 1/3 |
| 3W Oil | event (event_f1) | GRU + KD (soft labels + CLIP features) | -0.050 | 1/3 |
| 3W Oil | event (event_f1) | GRU + KD (probe soft labels) | -0.005 | 2/3 |
| 3W Oil | event (event_f1) | GRU + KD (zero-shot CLIP) | +0.024 | 1/3 |
| 3W Oil | event (event_f1) | GRU, window head from teacher only (beta=0) | -0.001 | 2/3 |
| 3W Oil | point (auprc) | GRU + KD (learned prompts, 2 views, confidence-weighted) | +0.001 | 1/3 |
| 3W Oil | point (auprc) | GRU + KD (learned prompts, 2 views) | +0.002 | 3/3 |
| 3W Oil | point (auprc) | GRU + KD (soft labels + CLIP features) | -0.002 | 1/3 |
| 3W Oil | point (auprc) | GRU + KD (probe soft labels) | +0.003 | 3/3 |
| 3W Oil | point (auprc) | GRU + KD (zero-shot CLIP) | -0.000 | 1/3 |
| 3W Oil | point (auprc) | GRU, window head from teacher only (beta=0) | -0.003 | 0/3 |
| 3W Oil | window (auprc) | GRU + KD (learned prompts, 2 views, confidence-weighted) | -0.007 | 0/3 |
| 3W Oil | window (auprc) | GRU + KD (learned prompts, 2 views) | -0.007 | 0/3 |
| 3W Oil | window (auprc) | GRU + KD (soft labels + CLIP features) | -0.005 | 1/3 |
| 3W Oil | window (auprc) | GRU + KD (probe soft labels) | -0.000 | 1/3 |
| 3W Oil | window (auprc) | GRU + KD (zero-shot CLIP) | -0.003 | 1/3 |
| 3W Oil | window (auprc) | GRU, window head from teacher only (beta=0) | -0.012 | 0/3 |

## Daphnet: primary assessment (v1) vs this revision (v2)

| Metric | v1 | v2 |
|---|---|---|
| Teacher window AUPRC | 0.029 | 0.401 |
| GRU (labels) window AUPRC | 0.280 | 0.627 |
| GRU+KD window AUPRC | 0.394 | 0.592 |
| GRU+KD point AUPRC | 0.307 | 0.477 |
| GRU+KD event F1 | 0.095 | 0.252 |

Air Quality (synthetic injected labels) is excluded from this benchmark for now; its runs are kept in `experiments/excluded_airquality/`.

## Caveat on MetroPT-3 (see `metropt/AUDIT.md`)

No leakage was found (disjoint sequences and events; no near-duplicate windows). The test set has only two failures and they are visually obvious: the window mean of one sensor (DV_pressure, chosen on training data) already reaches test window AUPRC 0.917. Per event, the single-view learned-prompt teacher scores 1.000 on failure 3 and 0.498 on failure 4. Block-bootstrap 95% intervals: learned prompts 0.50-1.00, probe 0.09-1.00. Read MetroPT numbers as evidence of generalisation from one subtle training failure to two obvious ones, not as near-perfect detection.
