# Methodology (v2.1) and response to the primary-assessment review

This document states exactly what is trained, on which targets, and how it is
evaluated. Numbers come from `experiments/benchmark_v2/RESULTS.md` and
`experiments/benchmark_v2/VERIFICATION.md`, which `scripts/finalize_benchmark.py`
and `scripts/verify_results.py` regenerate from the saved runs.

**Scope.** Four datasets with real labels: Credit Card, Daphnet FoG, Petrobras 3W
and MetroPT-3. Air Quality has only synthetic injected labels and is excluded
for now; its runs and label audit are kept in `experiments/excluded_airquality/`.

## 1. Pipeline at a glance

```
 multivariate window X (L x F, standardised with train-only statistics)
   |                                                   |
   | render, label-free:                               | numeric tensor
   |   view 1 heatmap (channels x time, clipped +-4)    v
   |   view 2 spectrogram (channel x frequency x time)  GRU student (conv front-end -> BiGRU)
   v                                                 |            |             |
 frozen CLIP ViT-B/32 image encoder -> e1, e2      window head  step head   embed head (train only)
   |                                                  z_w (2)     z_s (L)       g (512)
   |-> zero-shot: hand-written prompt ensemble
   |-> linear probe on [e1, e2]               (cross-fitted)
   |-> learned prompts (CoOp) on e1 and e2    (cross-fitted)   <- final teacher
   v
 teacher soft label p_T, embedding e = norm(e1 + e2)  -- training targets only --
```

At inference only the GRU runs (94k–101k parameters, about 1/1500 of CLIP);
it scores windows roughly 170x faster than the CLIP teacher on this CPU.

## 2. Teacher (all CLIP weights frozen)

* **Backbone.** `openai/clip-vit-base-patch32`; `requires_grad=False` everywhere,
  checked at runtime.
* **View 1, heatmap.** Each standardised channel is clipped to [-4, 4] and drawn
  as one row of a viridis heatmap. v1 used one global colour range (about
  -95..108 on Daphnet), so most windows were near-uniform images; the clipped
  rendering raised probe AUPRC on a Daphnet validation sample from 0.666 to
  0.780 (`render_ablation_daphnet.json`).
* **View 2, spectrogram.** Short-time Fourier magnitude of each channel
  (Hann window, fixed log scale, no per-window normalisation), stacked
  channel x frequency vertically. Rhythmic events (freezing-of-gait tremor at
  3–8 Hz, slugging, compressor cycling) are clearer in frequency than in
  amplitude. `src/teacher/render.py`.
* **Zero-shot teacher.** Image embedding vs. the mean text embedding of four
  hand-written normal and four anomalous prompts, times CLIP's logit scale.
* **Linear-probe teacher.** Logistic regression on the frozen embeddings
  (heatmap only, or heatmap + spectrogram concatenated).
* **Prompt-learning teacher (CoOp; final teacher).** The prompt
  `[SOS] v1 ... v14 normal|anomalous . [EOS]` has 14 learnable context vectors
  (7,168 numbers) initialised from the phrase *"a heatmap of a multivariate
  sensor time series that looks"*. They pass through CLIP's frozen text
  encoder; p(class | image) = softmax(100 · cos(e_image, f_text(class))).
  With two views, the class logits are averaged over the heatmap and
  spectrogram embeddings with the same learned prompts. Only the context
  vectors are trained (Adam, 12 epochs, cross-entropy on training windows).
  A unit test checks that, before training, the learnable prompt reproduces
  CLIP's own text features exactly. `src/teacher/prompt_learning.py`.
* **Cross-fitting (probe and prompts).** Soft labels for training windows are
  out-of-fold: the time-ordered windows are cut into blocks of 32, assigned to
  5 folds round-robin, and each block is labelled by a teacher fitted on the
  other folds. *Fix in v2.1:* v2.0 used 5 contiguous folds, which put
  MetroPT's single training failure inside one fold, so its soft labels there
  came from a teacher that had never seen a failure (out-of-fold AUPRC 0.03).
  The interleaved blocks spread an event over several folds while keeping
  near-duplicate overlapping windows mostly together. A final teacher fitted on
  all training windows labels validation/test windows (reporting only).

## 3. Student and training objective

`src/student/model.py`: Conv1d(F→64, k=5) → Conv1d(64→64, k=5, stride=d) →
BiGRU(64) → three heads. *d* = 4 for Daphnet (256-step windows), 2 for
3W/MetroPT, 1 for Credit Card; step logits are linearly upsampled back to L.

### Which target supervises which head

| Head | Output | Target | Loss term | Weight |
|---|---|---|---|---|
| step head | z_s ∈ R^L | ground-truth timestep labels y_step | weighted BCE | 1 |
| window head | z_w ∈ R^2 | ground-truth window label y_w = max(y_step) | cross-entropy | β = 1 |
| window head | z_w ∈ R^2 | teacher soft label p_T (cross-fitted) | w_i · T²·KL(p_T^(T) ‖ softmax(z_w/T)) | λ_KD = 1 |
| embed head | g(h̄) ∈ R^512 | frozen CLIP image embedding e | 1 − cos(g, e) | λ_feat = 0.5 |

```
L = L_step + β·L_window + λ_KD·mean_i(w_i·KL_i) + λ_feat·L_feat
```

* **Confidence weighting** (`gru_kd_coop`): w_i = p_T(y_i), the teacher's
  probability for the window's true class. Windows the teacher gets wrong are
  distilled less, and a teacher that is often wrong contributes less overall.
  Without weighting, w_i = 1.
* `pos_weight` of the step BCE is `clip(sqrt(neg/pos), 1, 25)` from the training labels.
* L_KD and L_feat are averaged only over teacher-scored training windows (up to 8,000).
* Teacher knowledge reaches the **window head** through L_KD and the **shared
  encoder** through L_feat, which also feeds the step head. At inference it
  reaches localization through the validation-tuned window gating (section 4).

### Worked example (one window, T = 2)

```
ground truth             y_window = 1,   y_step = [0, 0, 1, 1]
teacher                  p_T = [0.20, 0.80]  -> softened at T=2: [0.333, 0.667]
student window logits    z_w = [0.10, 0.90]  -> softmax: [0.310, 0.690]
                                             -> softened at T=2: [0.401, 0.599]

L_window = -log 0.690                                                  = 0.371
KL       = 0.333 ln(0.333/0.401) + 0.667 ln(0.667/0.599)               = 0.0097
L_KD     = T² · KL = 4 · 0.0097 = 0.039   (confidence-weighted: × p_T(y=1) = 0.8 → 0.031)
L_step   = mean over the 4 steps of weighted BCE(z_s[t], y_step[t])
L_feat   = 1 - cos(g(h̄), e_CLIP)
```

### Variants (identical splits, identical initial weights per seed, seeds 0/1/2)

| Variant | L_step | L_window (GT) | L_KD | L_feat | Teacher |
|---|---|---|---|---|---|
| `gru_gt` | ✓ | ✓ | – | – | none |
| `gru_kd_zeroshot` | ✓ | ✓ | ✓ | – | zero-shot CLIP (heatmap) |
| `gru_kd_probe` | ✓ | ✓ | ✓ | – | probe (heatmap) |
| `gru_kd_full` | ✓ | ✓ | ✓ | ✓ | probe (heatmap) |
| `gru_teacher_only` | ✓ | – (β = 0) | ✓ | ✓ | probe (heatmap) |
| `gru_kd_coop_plain` | ✓ | ✓ | ✓ | ✓ | learned prompts, 2 views |
| `gru_kd_coop` (**proposed**) | ✓ | ✓ | ✓ confidence-weighted | ✓ | learned prompts, 2 views |

Selection: best validation mean of window and step AUPRC (the validation loss
never contains teacher terms). Training: AdamW, learning rate 2e-3, weight
decay 1e-4, batch 128, at most 20 epochs, patience 5, ReduceLROnPlateau.

A variance-reduced Daphnet configuration (weight EMA, gain/noise augmentation,
learning rate 1e-3) was also run. The base configuration had the higher mean
validation AUPRC (0.834 vs. 0.830), so it is the one reported
(`daphnet/config_selection.json`); the tuned run is a sensitivity analysis in
`experiments/benchmark_v2_daphnet_tuned/`.

## 4. Localization and evaluation protocol

1. Average per-step scores over overlapping windows.
2. Gate: `score = step × window_prob^α`, α ∈ {0, 0.5, 1}.
3. Smooth with a centred moving average k ∈ {1, L/16, L/8, L/4}.
4. Threshold maximising validation point-F1.
5. Drop predicted runs shorter than `min_len` (maximising validation event-F1).

α and k are chosen by validation point AUPRC. Nothing is tuned on test, and no
point-adjustment is applied. Every table reports AUPRC with **prevalence**
(a random detector's AUPRC equals the prevalence).

## 5. Datasets

| Dataset | Split | Labels |
|---|---|---|
| Credit Card | chronological 60/20/20 | native fraud flag (0.13% of test transactions) |
| Daphnet FoG | subject-disjoint (train 7 subjects, val S02, test S04+S07) | native freeze annotation |
| 3W | instance-disjoint, 364 instances (real first, no hand-drawn), 10 s averages | native class ≠ 0 |
| MetroPT-3 | failure 1 train, 2 val, 3+4 test; normal blocks chronological | four published failure intervals |

## 6. Is the improvement real?

`scripts/verify_results.py` writes `VERIFICATION.md`:

* every saved test AUPRC is recomputed with scikit-learn from the saved
  predictions (difference 0.0); scores ×100 give the same AUPRC; random
  scores and shuffled labels give the prevalence;
* the v1 saved models re-scored on the identical Daphnet test split reproduce
  the v1 numbers (0.056 for the v1 UI checkpoint; 0.280 / 0.394 for the v1
  controlled study), then each v2 change is added one at a time;
* partitions share no subject/instance/event; scaler, teachers, thresholds and
  localization use training or validation data only.

**MetroPT-3 caveat.** Its test set has two visually obvious failures; a one-sensor rule chosen on training data reaches 0.917 window AUPRC and the teacher's interval is wide (0.50–1.00). Details: `experiments/benchmark_v2/metropt/AUDIT.md` (`scripts/audit_metropt.py`).

## 7. Response to the review

| # | Review point | What changed | Evidence |
|---|---|---|---|
| 1 | Do teacher scores help beyond ground truth alone? | Paired GT-only vs KD students: same split, same initial weights, 3 seeds, 4 datasets | `paired_vs_gt.csv`, RESULTS.md |
| 2 | Training objective unclear | One implementation (`TotalLoss`), equation, head/target table, worked example | `src/student/losses.py` |
| 3 | Low AUPRC on several datasets | Teacher rendering fix, probe and learned-prompt teachers, spectrogram view, stronger student, AUPRC-based selection, tuned localization | RESULTS.md, VERIFICATION.md |
| 4 | Which target supervises each head | Table in section 3 | this file |
| 5 | Frozen CLIP vs GRU-GT vs GRU-KD with prevalence; verify Air Quality labels; target one weak dataset | All on identical test windows; prevalence everywhere; Air Quality labels audited and found synthetic → excluded; Daphnet attribution ladder | RESULTS.md, VERIFICATION.md |
| 6 | Literature review and motivation | Rewritten report sections 1–2 | `final_ie643_report/main.tex` |
| 7 | Soft labels, localization, more experiments | Five teachers, seven students, confidence-weighted KD, gated localization, 3 seeds | RESULTS.md |
| 8 | Novelty via comparisons and ablations | Learned prompts + multi-view teacher; Isolation Forest and gradient-boosted-tree baselines; component ablations | RESULTS.md |
