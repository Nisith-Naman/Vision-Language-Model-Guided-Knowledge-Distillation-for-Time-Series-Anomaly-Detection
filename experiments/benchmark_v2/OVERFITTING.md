# Overfitting audit

Window AUPRC per split (prevalence in brackets). A large train-minus-test gap with a low test score means overfitting; a high test score with a small gap means the model generalises. Teacher train scores are out-of-fold. Students: deployed checkpoint (best validation seed).

## Credit Card

Prevalence: train 0.098, val 0.058, test 0.074.

| Model | Train | Val | Test | Train − Test |
|---|---|---|---|---|
| GRU, labels only (seed 2) | 0.792 | 0.802 | 0.826 | -0.034 |
| GRU + KD (proposed) (seed 1) | 0.806 | 0.785 | 0.791 | +0.015 |
| Teacher: linear probe (train = out-of-fold) | 0.462 | 0.193 | 0.217 | +0.245 |
| Teacher: learned prompts, 2 views (train = out-of-fold) | 0.321 | 0.086 | 0.124 | +0.197 |

Training dynamics of the base-configuration runs (all seeds): best epoch, and validation AUPRC lost between the best and the last epoch trained (protected by early stopping):

- gru_gt: best epochs [2, 2, 2]; val-AUPRC drop after best [0.024, 0.033, 0.04]
- gru_kd_coop: best epochs [2, 2, 2]; val-AUPRC drop after best [0.027, 0.032, 0.048]

## Daphnet FoG

Prevalence: train 0.171, val 0.189, test 0.048.

| Model | Train | Val | Test | Train − Test |
|---|---|---|---|---|
| GRU, labels only (seed 1) | 0.862 | 0.842 | 0.641 | +0.221 |
| GRU + KD (proposed) (seed 1) | 0.828 | 0.843 | 0.634 | +0.195 |
| Teacher: linear probe (train = out-of-fold) | 0.760 | 0.785 | 0.542 | +0.217 |
| Teacher: learned prompts, 2 views (train = out-of-fold) | 0.645 | 0.665 | 0.401 | +0.244 |

Training dynamics of the base-configuration runs (all seeds): best epoch, and validation AUPRC lost between the best and the last epoch trained (protected by early stopping):

- gru_gt: best epochs [2, 1, 3]; val-AUPRC drop after best [0.129, 0.046, 0.144]
- gru_kd_coop: best epochs [1, 1, 1]; val-AUPRC drop after best [0.065, 0.024, 0.108]

## 3W Oil

Prevalence: train 0.654, val 0.704, test 0.620.

| Model | Train | Val | Test | Train − Test |
|---|---|---|---|---|
| GRU, labels only (seed 0) | 0.987 | 0.993 | 0.987 | -0.000 |
| GRU + KD (proposed) (seed 0) | 0.982 | 0.989 | 0.983 | -0.001 |
| Teacher: linear probe (train = out-of-fold) | 0.972 | 0.987 | 0.992 | -0.019 |
| Teacher: learned prompts, 2 views (train = out-of-fold) | 0.940 | 0.976 | 0.971 | -0.031 |

Training dynamics of the base-configuration runs (all seeds): best epoch, and validation AUPRC lost between the best and the last epoch trained (protected by early stopping):

- gru_gt: best epochs [9, 5, 6]; val-AUPRC drop after best [0.005, 0.129, 0.082]
- gru_kd_coop: best epochs [4, 3, 3]; val-AUPRC drop after best [0.015, 0.007, 0.038]

## MetroPT-3

Prevalence: train 0.007, val 0.009, test 0.062.

| Model | Train | Val | Test | Train − Test |
|---|---|---|---|---|
| GRU, labels only (seed 1) | 1.000 | 1.000 | 0.856 | +0.144 |
| GRU + KD (proposed) (seed 0) | 1.000 | 1.000 | 0.899 | +0.101 |
| Teacher: linear probe (train = out-of-fold) | 0.418 | 1.000 | 0.931 | -0.512 |
| Teacher: learned prompts, 2 views (train = out-of-fold) | 0.842 | 0.894 | 0.998 | -0.156 |

Training dynamics of the base-configuration runs (all seeds): best epoch, and validation AUPRC lost between the best and the last epoch trained (protected by early stopping):

- gru_gt: best epochs [1, 1, 6]; val-AUPRC drop after best [0.004, 0.001, 0.098]
- gru_kd_coop: best epochs [1, 1, 2]; val-AUPRC drop after best [0.039, 0.002, 0.058]
