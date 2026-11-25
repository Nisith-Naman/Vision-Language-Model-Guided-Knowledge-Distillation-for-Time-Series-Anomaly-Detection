# Generalisation study: base vs. regularized training

Regularized = weight EMA (0.99), gain/noise augmentation (0.15 / 0.05), learning rate 1e-3, dropout 0.3, patience 6. Same splits, teachers, seeds and initial-weight seeds as the base runs. The choice per dataset and model uses **validation AUPRC only**; test numbers are reported afterwards.

| Dataset | Model | Val AUPRC base → reg | Test window AUPRC base → reg | Test point AUPRC base → reg | Seed std (window) base → reg | Train−test gap base → reg | Selected |
|---|---|---|---|---|---|---|---|
| Credit Card | GRU, labels only | 0.770 → 0.760 | 0.823 → 0.828 | 0.787 → 0.777 | 0.005 → 0.005 | -0.034 → +0.037 | **base** |
| Credit Card | GRU + KD (proposed) | 0.766 → 0.756 | 0.801 → 0.805 | 0.775 → 0.776 | 0.011 → 0.006 | +0.015 → +0.020 | **base** |
| Daphnet FoG | GRU, labels only | 0.831 → 0.835 | 0.627 → 0.659 | 0.519 → 0.535 | 0.058 → 0.017 | +0.256 → +0.221 | **regularized (near-tie: val Δ +0.0045)** |
| Daphnet FoG | GRU + KD (proposed) | 0.838 → 0.840 | 0.592 → 0.655 | 0.477 → 0.549 | 0.068 → 0.030 | +0.294 → +0.195 | **regularized (near-tie: val Δ +0.0024)** |
| 3W Oil | GRU, labels only | 0.989 → 0.988 | 0.985 → 0.985 | 0.978 → 0.979 | 0.003 → 0.007 | -0.000 → -0.001 | **base (near-tie: val Δ -0.0011)** |
| 3W Oil | GRU + KD (proposed) | 0.984 → 0.988 | 0.978 → 0.983 | 0.978 → 0.986 | 0.004 → 0.000 | +0.003 → -0.001 | **regularized (near-tie: val Δ +0.0036)** |
| MetroPT-3 | GRU, labels only | 1.000 → 1.000 | 0.751 → 0.812 | 0.524 → 0.821 | 0.137 → 0.051 | +0.403 → +0.144 | **regularized (near-tie: val Δ +0.0001)** |
| MetroPT-3 | GRU + KD (proposed) | 1.000 → 1.000 | 0.876 → 0.892 | 0.889 → 0.882 | 0.015 → 0.011 | +0.110 → +0.101 | **regularized (near-tie: val Δ +0.0004)** |

**Reading the selection.** Validation separates the two configurations only weakly: on MetroPT it is saturated (both ≈ 1.000; differences ≈ 0.0001, because validation holds a single failure) and on Daphnet and 3W the differences are below 0.005. These choices are therefore near-ties under a pre-set rule, not validated wins. What the regularized recipe does show consistently on test is lower seed variance and a smaller train−test gap where the base models overfit (Daphnet, MetroPT labels-only GRU), which is its purpose. Earlier, the same recipe lost to the base configuration on Daphnet when validation AUPRC was averaged over all five student variants (0.830 vs 0.834); that comparison fixed the configuration of the main ablation table.

Deployment: the selected configuration's best-validation-seed checkpoint is served by the interface (`checkpoints/v2_<dataset>_<model>`). The main ablation tables in RESULTS.md use the base configuration for every variant, so all variants remain directly comparable.

Deployed regularized checkpoints: daphnet/gru_gt, daphnet/gru_kd_coop, oil3w/gru_kd_coop, metropt/gru_gt, metropt/gru_kd_coop (previous ones kept as `checkpoints/v2base_*`).
## Does distillation still help against a regularized baseline?

Paired differences (proposed student − labels-only GRU, same seeds and splits), base vs. regularized training:

| Dataset | Level | Base training: labels-only → KD (Δ, wins) | Regularized training: labels-only → KD (Δ, wins) |
|---|---|---|---|
| MetroPT-3 | window AUPRC | 0.751 → 0.876 (+0.124, 3/3) | 0.812 → 0.892 (+0.080, 3/3) |
| MetroPT-3 | point AUPRC | 0.524 → 0.889 (+0.365, 3/3) | 0.821 → 0.882 (+0.061, 3/3) |
| MetroPT-3 | event F1 | 0.250 → 0.722 (+0.472, 2/3) | 0.577 → 0.820 (+0.242, 2/3) |
| Daphnet FoG | window / point AUPRC | −0.035 / −0.042 | −0.004 / +0.015 |
| 3W Oil | window / point AUPRC | −0.007 / +0.001 | −0.002 / +0.007 |
| Credit Card | window / point AUPRC | −0.022 / −0.012 | −0.022 / −0.001 |

**Conclusion.** Most of the large MetroPT gain under base training came from a labels-only model that overfitted
its single training failure. Explicit regularization recovers much of it. Distillation still adds a consistent
gain on MetroPT (+0.06 to +0.08 AUPRC on every seed) and remains neutral elsewhere. Out of sample, with the
deployed (regularized) models, both MetroPT students find 10/12 hidden real failures; the distilled student has
higher point AUPRC (0.717 vs 0.672) and event F1 (0.546 vs 0.424). With the base checkpoints the result was 10/12
vs 0/12 (`OUT_OF_SAMPLE_base_checkpoints.md`).
