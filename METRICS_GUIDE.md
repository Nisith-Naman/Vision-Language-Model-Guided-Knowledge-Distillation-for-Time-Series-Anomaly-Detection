# Metrics guide: what every number in the app means

This guide covers every page of the Streamlit app (`app.py`) and every metric in the report. All metrics are
computed by `src/evaluation/postprocess.py`, the same code that produced the benchmark tables. Page 5
(Evaluation) re-derives each one in the open and checks it against `experiments/*/metrics.csv`.

## 1. Basic terms

| Term | Meaning |
|---|---|
| Timestep / point | one row of the series (one transaction, one 1/64 s sample, one 10 s or 1 min average). |
| Window | L consecutive timesteps, cut every `stride` steps. Credit Card L = 64, Daphnet 256, 3W and MetroPT 120. |
| Window label `y_win` | 1 if *any* timestep in the window is anomalous (`max_t y_step`). |
| Event | a maximal run of consecutive anomalous timesteps (one fraud, one freezing episode, one failure). |
| Prevalence | the share of anomalous items (windows, timesteps). **It equals the AUPRC of a random detector**, so every AUPRC is shown next to it. |
| Validation split | used only to choose thresholds and post-processing settings. |
| Test split | scored once, with those settings fixed. |

## 2. The three levels of evaluation

| Level | Question | Score used | Decision |
|---|---|---|---|
| **Window** | Is this window anomalous? | Student: window-head probability `p_win`. Teacher: `p_T`. | score ≥ τ_w |
| **Point** | Is this timestep anomalous? | Step-head scores averaged over the overlapping windows, gated by `p_win^α`, smoothed over k steps (teacher: its window score spread over the window). | score ≥ τ_p, then runs shorter than m are removed |
| **Event** | Was each anomalous interval found? | the point decisions | a true event is *detected* if any of its steps is flagged; a predicted event is *correct* if it overlaps a true anomalous step |

**Choosing the threshold** (τ_w, τ_p) happens on validation only:

- **5 or more validation events:** τ is the value that maximises F1 on validation.
- **Fewer than 5 events** (only MetroPT, which has 1): τ is the 99.9th percentile of the scores of the *normal*
  validation items, i.e. a 0.1 % false-alarm rate. An F1-optimal threshold fitted to one event varied from
  0.004 to 0.35 across seeds.
- **α, k and m** are also chosen on validation:
  - α ∈ {0, 0.5, 1} and k ∈ {1, L/16, L/8, L/4} maximise validation point AUPRC;
  - m maximises validation event F1.

## 3. The metrics

Write TP, FP, FN and TN for true positives, false positives, false negatives and true negatives, counted at the
chosen threshold.

| Metric | Formula | Range and reading |
|---|---|---|
| **Precision** | TP / (TP + FP) | Of everything flagged, the share that is truly anomalous. Low precision means false alarms. |
| **Recall** | TP / (TP + FN) | Of all true anomalies, the share flagged. Low recall means missed anomalies. |
| **F1** | 2·P·R / (P + R) | Balances the two. 0 if nothing correct is flagged. Depends on the threshold. |
| **AUPRC** (average precision) | Σₖ (Rₖ − Rₖ₋₁)·Pₖ over all cut-offs of the score | Threshold-free quality of the *ranking*. 1 = every anomaly scores above every normal item; prevalence = random. **Primary metric.** |
| **ROC-AUC** | P(random anomaly scores above random normal item) | 0.5 = random, 1 = perfect. Looks optimistic when anomalies are rare, so it is secondary. |
| **Event precision** | correct predicted events / predicted events | Share of alarms (intervals) that hit a real anomaly. |
| **Event recall** | detected true events / true events | Share of anomalous episodes found at least partly. |
| **Event F1** | harmonic mean of the two | Episode-level usefulness of the alarms. |

**Why AUPRC, not accuracy or ROC-AUC.** With 0.1–6 % anomalies, a detector that never fires has 94–99.9 %
accuracy. AUPRC falls to the prevalence for such a detector, so it shows real progress.

**Why a model can have high AUPRC but F1 = 0.** AUPRC ignores the threshold. F1 is 0 when the threshold flags
nothing, for example a 0.5 cut-off on a model whose anomaly probabilities are all below 0.5. That is why every
threshold here is tuned on validation.

**No point-adjustment.** Some papers mark a whole true event as detected when one step is flagged and then count
every step as a true positive, which inflates point F1 even for random scores. We do not. Event credit appears
only in the event metrics.

## 4. Training metrics (page 4)

| Quantity | Meaning |
|---|---|
| **Step BCE** | Weighted binary cross-entropy of the step head against `y_step`. Positive steps are weighted by w = clip(√(n_neg/n_pos), 1, 25). |
| **Window CE** | Cross-entropy of the window head against `y_win`. |
| **KD term** | T²·mean(wᵢ·KL(p_T^(T) ‖ p_win^(T))), with T = 2. It pulls the window head towards the teacher's softened probability. wᵢ = the teacher's probability of the true class, so wrong teacher labels count less. Only on teacher-scored training windows. |
| **Feature term** | 1 − cos(g, e): how far the embedding head's guess is from CLIP's (view-averaged) image embedding. |
| **Total loss** | step + β·window + λ_KD·KD + λ_feat·feature, with β = 1, λ_KD = 1, λ_feat = 0.5. The labels-only GRU uses λ_KD = λ_feat = 0. |
| **Validation loss** | Label terms only (step + window), so both models are judged alike. |
| **Validation AUPRC** | Mean of validation window AUPRC and step AUPRC after each epoch. **The epoch with the highest value is kept** (early stopping, patience 5; 6 when regularized). |
| **Best epoch** | The kept epoch, shown as a dashed line. |

**Reading the curves.**

- Training loss falling while validation AUPRC falls means overfitting; early stopping keeps the best epoch.
- A KD term that stays well above 0 means the student disagrees with the teacher on many windows. That is
  expected where the teacher is weak, such as Credit Card.

## 5. Teacher metrics (page 3)

| Quantity | Meaning |
|---|---|
| `cos(e, t)` | Cosine similarity between CLIP's image embedding of a rendered window and a text embedding. Both are unit vectors, so this is their dot product. |
| Logit `z_c = 100·cos(e, t_c)` | CLIP's own scale. With two views the logits are averaged. |
| `p_T = softmax(z)` | The teacher's probability that the window is normal or anomalous. |
| Out-of-fold train AUPRC | Quality of the soft labels the student actually received. Each comes from prompts fitted on other folds. |
| Val / test AUPRC, ROC-AUC | The teacher's own window-level detection quality, next to prevalence. |

## 6. Checks shown on the pages

| Check | Where | Passes when |
|---|---|---|
| Hand z-score = model input | page 2 | max difference < 10⁻³ |
| Window = rows s…s+L−1 of the scaled sequence | page 2 | identical |
| Live CLIP embedding = cached embedding | page 3 | cosine > 0.999 |
| Live teacher probability = cached probability | page 3 | difference < 10⁻³ |
| Hand-computed loss = `TotalLoss` | page 4 | difference < 10⁻⁴ |
| Page metrics = benchmark `metrics.csv` | page 5 | difference < 10⁻⁶ |

## 7. Results pages (6 and 7)

- **Page 6** shows point-level metrics over the timesteps of the CSV you scored, plus throughput (windows per
  second on this CPU).
- **Train − test gap** (overfitting tab): train AUPRC minus test AUPRC.
  - A large gap with a low test score means memorisation.
  - A gap shared by every model, including the out-of-fold teacher, means the test data differs (Daphnet: new
    people).
- **Out-of-sample** results come from label-free generated series graded with a hidden answer key:
  - "found 10/12" = hidden real anomalies with at least one flagged step;
  - generic shapes are reported separately.
