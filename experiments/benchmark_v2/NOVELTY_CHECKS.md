# Functional checks of the novelty components

## 1. Prompt learning (CoOp)

- Trainable tensors during prompt learning: ['ctx'] (7168 numbers).
- CLIP state-dict hash before training 12f1a938acc3449f, after 12f1a938acc3449f: **unchanged**.
- Training loss per epoch (MetroPT, 2 views): [0.0743, 0.0331, 0.0312] (decreasing).

Learned prompt vs. its own untrained initial phrase (validation and test window AUPRC; same frozen CLIP, same two views):

| Dataset | Initial phrase (untrained) val / test | Learned prompts val / test |
|---|---|---|
| creditcard | 0.080 / 0.101 | 0.086 / 0.124 |
| daphnet | 0.218 / 0.064 | 0.665 / 0.401 |
| oil3w | 0.943 / 0.871 | 0.976 / 0.971 |
| metropt | 0.007 / 0.083 | 0.894 / 0.998 |

## 2. Multi-view (heatmap + spectrogram)

| Dataset | mean cos(heatmap, spectrogram) | probe on heatmap only, val AP | probe on spectrogram only, val AP | learned prompts 1 view → 2 views, test AP |
|---|---|---|---|---|
| creditcard | 0.899 | 0.064 | 0.073 | 0.137 → 0.124 |
| daphnet | 0.890 | 0.391 | 0.454 | 0.531 → 0.401 |
| oil3w | 0.821 | 0.949 | 0.939 | 0.974 → 0.971 |
| metropt | 0.910 | 0.936 | 0.321 | 0.989 → 0.998 |

Cosines of 0.82-0.91 (views related but not identical) and a spectrogram-only probe above prevalence show that the second view adds information; whether it helps the final teacher is dataset-dependent (last column).

## 3. Confidence-weighted KD on a real batch (Daphnet, 256 training windows)

- KD term from the training code 1.038916; hand-computed mean(p_T(y)·T²·KL) 1.038916: **match**.
- Mean weight where the teacher is right: 0.906 (218 windows); where it is wrong: 0.273 (38 windows).
