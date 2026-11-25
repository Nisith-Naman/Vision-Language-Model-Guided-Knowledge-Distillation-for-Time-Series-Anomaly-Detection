# The whole system on a toy example (16 rows, 2 sensors)

The real datasets have thousands of rows and up to 29 sensors, so the numbers in the app are hard to follow.
This page runs **the same steps** on a toy file small enough to do with a calculator.
Every number below is printed by `scripts/toy_example.py`, which uses the project's own functions
(interpolation, windowing, image rendering, `TotalLoss`, `select_threshold`, `aggregate`, `event_scores`).

```powershell
.venv\Scripts\python.exe scripts\toy_example.py
```

Each section names the app page that shows the same step on the real data.

---

## Part A · Data preparation (app page 2)

### Step 1 · CSV → matrix

A machine with two sensors, one row per minute. The last column says whether a human marked the row as a fault.

| row | temp | pressure | label |   | row | temp | pressure | label |
|---|---|---|---|---|---|---|---|---|
| 0 | 18 | 95 | 0 | | 8 | 20 | 100 | 0 |
| 1 | 22 | 105 | 0 | | 9 | 22 | 105 | 0 |
| 2 | *missing* | 95 | 0 | | 10 | 18 | 95 | 0 |
| 3 | 22 | 105 | 0 | | 11 | 20 | 100 | 0 |
| 4 | 18 | 105 | 0 | | 12 | **30** | **110** | **1** |
| 5 | 18 | 95 | 0 | | 13 | **32** | **115** | **1** |
| 6 | 22 | 105 | 0 | | 14 | 20 | 100 | 0 |
| 7 | 18 | 95 | 0 | | 15 | 18 | 95 | 0 |

The sensor columns become a matrix **X** with 16 rows (time) × 2 columns (sensors), so F = 2.
The label column becomes a vector **y** = [0,0,0,0,0,0,0,0, 0,0,0,0,1,1,0,0].
Rows 12–13 are one **event**, a run of consecutive anomalous rows: the machine overheated for 2 minutes.

### Step 2 · Split

The split is chronological. **Rows 0–7 are for training and rows 8–15 for testing.** The real system also
keeps a validation part in the middle; it is cut exactly like the test part.

A row never belongs to two parts, so the model is tested on rows it has never seen.

### Step 3 · Fill missing values

Row 2 has no temperature. It is filled by a straight line between its neighbours in time:
row 1 = 22 and row 3 = 22, so **row 2 = 22**.

### Step 4 · Standardise, using training rows only

The training temperatures are 18, 22, 22, 22, 18, 18, 22, 18.

- Mean: μ = 160 / 8 = **20**.
- Each value is ±2 from the mean, so the standard deviation is σ = **2**.

The training pressures are all 95 or 105, so μ = **100** and σ = **5**.

Every value becomes **z = (x − μ) / σ**, meaning "how many standard deviations from normal". The test rows become:

| test row | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|
| temp z = (x−20)/2 | 0 | 1 | −1 | 0 | **5** | **6** | 0 | −1 |
| pressure z = (x−100)/5 | 0 | 1 | −1 | 0 | **2** | **3** | 0 | −1 |
| label | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 |

The fault is already visible: the temperature is 5–6 σ above normal.

The test data is standardised with the training μ and σ. Fitting μ and σ on the test rows would leak test
information into the model.

### Step 5 · Cut windows (L = 4, stride 2)

The models never see a whole sequence. They see short **windows** of L rows that start every S rows.
The test part has 8 rows, so the windows start at 0, 2 and 4:

| window | rows (test index) | temp z | pressure z | step labels y_step | window label y_win = max |
|---|---|---|---|---|---|
| W0 | 0–3 | 0, 1, −1, 0 | 0, 1, −1, 0 | 0 0 0 0 | **0** |
| W1 | 2–5 | −1, 0, 5, 6 | −1, 0, 2, 3 | 0 0 1 1 | **1** |
| W2 | 4–7 | 5, 6, 0, −1 | 2, 3, 0, −1 | 1 1 0 0 | **1** |

- Each window is a small **L × F = 4 × 2 matrix**, the model's input.
- Windows overlap by L − S = 2 rows, so a row can appear in up to 2 windows.
- A window is labelled anomalous if **any** of its rows is anomalous.

In the real data L is 64–256 and F is 8–29, and there are thousands of windows. That is the only difference.

---

## Part B · The teacher (app page 3)

The teacher is CLIP, a large pretrained image-and-text model. Its weights are **frozen** (never changed).
Its job is to give every training window a **second opinion**, a probability p_T, to learn from.
The steps below follow window W1.

**1. Turn the numbers into a picture.**

- Clip each z to [−4, 4], so temp 5 and 6 both become 4.
- Map each value to a colour position (z + 4) / 8:

  | | step 1 | step 2 | step 3 | step 4 |
  |---|---|---|---|---|
  | temp | 0.375 | 0.5 | **1.0** | **1.0** |
  | pressure | 0.375 | 0.5 | 0.75 | 0.875 |

- Colour each cell with the viridis colour map (0 = dark purple, 1 = yellow). Stretch the 2 × 4 grid to a
  224 × 224 image, so each cell becomes a 112 × 56 pixel block.

The labels are **never** drawn into the image. A second picture, the **spectrogram**, shows how much each
sensor oscillates at each frequency.

**2. CLIP looks at each picture.** It turns each image into 512 numbers, e. It compares e with two learned
text descriptions, "… normal" and "… anomalous", using cosine similarity × 100. These two scores are the
*logits*. In the toy:

| view | normal | anomalous |
|---|---|---|
| heatmap | 21 | 23 |
| spectrogram | 22 | 22 |
| **mean over views** | **21.5** | **22.5** |

**3. Softmax turns logits into a probability:**

p_T(anomalous) = e^22.5 / (e^21.5 + e^22.5) = 1 / (1 + e^(−1)) = **0.731**

So the teacher says W1 is anomalous with probability 0.73.

> **What was trained?** Only the 14 "context" words in front of "normal / anomalous" (14 × 512 = 7,168
> numbers). Everything else in CLIP is frozen.
>
> **No cheating on training windows.** A window's own label could leak into its p_T. So the training windows
> are split into 5 folds: the prompts learned on 4 folds score the 5th fold (*cross-fitting*).

---

## Part C · The student (app page 4)

The student is a small network (~95k numbers) that reads the **raw z-scores**, not pictures. It is the model
that is actually deployed.

**1. Front-end and GRU.** Two 1-D convolutions look at short patterns, then a bidirectional GRU reads the
window left-to-right and right-to-left. For each time step it outputs a vector h_t (128 numbers; 2 in the toy):

| step | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| h_t | (0, 0) | (0, 0.2) | (0.8, 0.4) | (1.2, 0.6) |

**2. Pooling.** The mean and maximum over time are put side by side:
u = [mean, max] = [0.5, 0.3, 1.2, 0.6].

**3. The three heads**, each answering a different question:

| head | question | toy computation | output |
|---|---|---|---|
| **Window head** | Is this window anomalous? | z_normal = 0.7 − 0.5·1.2 = 0.1;  z_anom = 0.1 + 0.5 + 0.5·0.6 = 0.9;  softmax | **p_win = 0.690** |
| **Step head** | Which rows are anomalous? | one logit per step: 3·h_t[0] − 1 = −1, −1, 1.4, 2.6;  sigmoid | **s_t = 0.27, 0.27, 0.80, 0.93** |
| **Embedding head** (training only) | What would CLIP see? | maps u to a vector g (512 numbers; 3 in the toy): g = (1, 2, 2) | compared with CLIP's e = (2, 1, 2) |

At test time only the window and step heads are used. The embedding head exists only to pass CLIP's "view"
into the student during training.

---

## Part D · The loss: how the student learns from labels **and** teacher (app page 4)

For W1, the true labels are y_step = 0 0 1 1 and y_win = 1. The settings are pos_weight = 3, T = 2, β = 1,
λ_KD = 1 and λ_feat = 0.5. The loss has four terms, and **lower is better**.

**1. Step loss** (step head vs y_step; weighted binary cross-entropy):

| step | s_t | label | loss |
|---|---|---|---|
| 1 | 0.269 | 0 | −ln(1 − 0.269) = 0.313 |
| 2 | 0.269 | 0 | 0.313 |
| 3 | 0.802 | 1 | −**3**·ln(0.802) = 0.661 |
| 4 | 0.931 | 1 | −3·ln(0.931) = 0.215 |
| **mean** | | | **L_step = 0.376** |

Anomalous steps are rare, so their errors count pos_weight = 3 times. Otherwise "always say normal" would
look good.

**2. Window loss** (window head vs y_win):
L_win = −ln p_win(anomalous) = −ln 0.690 = **0.371**.

**3. Distillation loss** (window head vs **teacher**):

1. Soften both opinions by dividing the logits by T = 2:
   - teacher: (21.5, 22.5) / 2 → softmax → (0.378, 0.622);
   - student: (0.1, 0.9) / 2 → (0.401, 0.599).
2. Measure the mismatch with the KL divergence:
   0.378·ln(0.378/0.401) + 0.622·ln(0.622/0.599) ≈ 0.0012. With three-decimal inputs you get about
   0.0011; the script, without rounding, gets 0.00118.
3. Weight it by how much to trust the teacher: w = the teacher's probability for the **true** class = 0.731.
4. Scale by T² = 4: L_KD = 4 × 0.731 × 0.00118 = **0.0035**.

Here the student already agrees with the teacher, so this term is tiny. When they disagree, it pulls the
window head towards the teacher.

**4. Feature loss** (embedding head vs CLIP embedding):

- cos(g, e) = (1·2 + 2·1 + 2·2) / (3 × 3) = 8/9 = 0.889.
- L_feat = 1 − 0.889 = **0.111**.

**Total:** L = 0.376 + 1·0.371 + 1·0.0035 + 0.5·0.111 = **0.806**.
`TotalLoss` in the project returns exactly this (0.8058).

**Labels-only baseline.** This is the same student with λ_KD = λ_feat = 0, so only terms 1 and 2 remain.
Comparing the two is how we measure what the teacher adds.

### Training = repeat this many times

1. **One update.** Take 128 windows (a *batch*) and compute the loss averaged over them. Backpropagation
   computes how each of the ~95k weights should change to lower it, and AdamW makes that small change.
2. **One epoch.** Repeat step 1 until every training window has been used once.
3. **After every epoch,** score the **validation** windows and compute their AUPRC (Part E). Keep a copy of
   the weights if this is the best validation AUPRC so far.
4. **Stop** when validation has not improved for 5 epochs (or after 20 epochs). The kept copy is the final
   model.

App page 4 shows these curves for the real runs: every loss term and the validation AUPRC per epoch, with
the best epoch marked. You can also train a student live and watch them grow.

---

## Part E · Evaluation (app page 5)

A model outputs **scores** (probabilities). To say "anomaly / normal" we need a **threshold τ**, and τ is
chosen on **validation**, never on test.

### Window level

**1. Choose τ on validation** (6 windows):

| score | 0.9 | 0.7 | 0.6 | 0.4 | 0.3 | 0.1 |
|---|---|---|---|---|---|---|
| label | 1 | 1 | 0 | 1 | 0 | 0 |

Try each cut and keep the best F1:

| predict anomaly if score ≥ | TP | FP | FN | precision | recall | F1 |
|---|---|---|---|---|---|---|
| 0.9 | 1 | 0 | 2 | 1.00 | 0.33 | 0.50 |
| 0.7 | 2 | 0 | 1 | 1.00 | 0.67 | 0.80 |
| 0.6 | 2 | 1 | 1 | 0.67 | 0.67 | 0.67 |
| **0.4** | **3** | **1** | **0** | **0.75** | **1.00** | **0.86** ← best |
| 0.3 | 3 | 2 | 0 | 0.60 | 1.00 | 0.75 |

Any cut between 0.3 and 0.4 gives this F1; the code picks the lowest, **τ = 0.301**.

Validation sets with fewer than 5 anomalous events are too small for this search. There, τ is set so that
only 0.1 % of the normal validation windows are flagged (a false-alarm budget).

**2. Apply τ to test** (8 windows). The threshold is fixed now, and test labels are only used to score:

| score | 0.95 | 0.80 | 0.55 | 0.45 | 0.25 | 0.20 | 0.15 | 0.05 |
|---|---|---|---|---|---|---|---|---|
| label | 1 | 1 | 0 | 1 | 1 | 0 | 0 | 0 |
| predicted (≥ 0.301) | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| outcome | TP | TP | FP | TP | FN | TN | TN | TN |

| metric | formula | toy |
|---|---|---|
| precision | TP / (TP + FP): of the alarms, how many were real | 3/4 = **0.75** |
| recall | TP / (TP + FN): of the real anomalies, how many we caught | 3/4 = **0.75** |
| F1 | 2·P·R / (P + R) | **0.75** |
| prevalence | share of anomalous windows (what random guessing scores in AUPRC) | 4/8 = 0.50 |
| **AUPRC** | sort by score; at each true anomaly, note the precision so far; average them | (1/1 + 2/2 + 3/4 + 4/5)/4 = **0.8875** |
| ROC-AUC | chance that a random anomaly scores above a random normal | 14/16 = **0.875** |

AUPRC and ROC-AUC **do not use τ**: they judge the ranking of the scores. Precision, recall and F1 **do** use τ.

**Why F1 can be 0 while AUPRC is high.** With τ = 0.99, nothing is ≥ 0.99, so nothing is flagged:
TP = 0, so P = R = F1 = 0. The scores are unchanged, so AUPRC is still 0.8875. This is what happened on
the old page 6: a fixed τ = 0.5 sat above all the model's test scores. The fix is to always choose τ on
validation.

### Point (timestep) level: where exactly?

Each row of the test sequence gets the **average** step score of the windows that contain it
(W0 covers rows 0–3, W1 rows 2–5 and W2 rows 4–7):

| test row | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| W0 step scores | 0.1 | 0.6 | 0.2 | 0.2 | | | | |
| W1 step scores | | | 0.27 | 0.27 | 0.80 | 0.93 | | |
| W2 step scores | | | | | 0.9 | 0.7 | 0.1 | 0.1 |
| **timeline = average** | 0.10 | **0.60** | 0.235 | 0.235 | **0.85** | **0.815** | 0.10 | 0.10 |
| label | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 |

With a threshold of 0.5 (chosen on validation), rows 1, 4 and 5 are flagged:
- Point precision = 2/3 = 0.667, recall = 2/2 = 1, F1 = **0.80**.
- Row 1 is a false alarm caused by one step in W0.

The validation-tuned clean-up steps remove such blips. Each is chosen on validation:
- **Minimum event length** (min_len = 2): flagged runs shorter than 2 rows are deleted. Row 1 goes, and
  point F1 = **1.00**.
- **Gating** (α = 1): each row's score is multiplied by the window head's average probability for that row.
  Row 1 becomes 0.6 × 0.1 = 0.06, because the window head thought W0 was normal. This is how the window head
  (the part trained by the teacher) helps localisation.
- **Smoothing** (k): a moving average over k rows.

### Event level: did we find each fault?

An **event** is a run of consecutive 1s. A true event counts as found if **any** flagged row overlaps it, and a
predicted run counts as correct if it overlaps **any** true event.

| | true events | predicted runs | found / correct | event P | event R | event F1 |
|---|---|---|---|---|---|---|
| before min_len | rows 4–5 | row 1; rows 4–5 | 1 of 1 / 1 of 2 | 0.50 | 1.00 | **0.67** |
| after min_len 2 | rows 4–5 | rows 4–5 | 1 of 1 / 1 of 1 | 1.00 | 1.00 | **1.00** |

We do **not** use *point-adjustment*, a common trick that counts every row of an event as found if one row is
found. It inflates point F1.

---

## How the toy maps to the real app

| toy step | real system | app page |
|---|---|---|
| 16 rows, 2 sensors | 10⁴–10⁶ rows, 8–29 sensors | 2 · Data pipeline, step 1 |
| μ = 20, σ = 2 | μ, σ per sensor from the training split (table on the page) | 2 · step 4, with a worked example on a real row |
| L = 4, stride 2, 3 windows | L = 64–256, stride 30–64, thousands of windows | 2 · step 5 (pick any window) |
| teacher p_T = 0.731 | the same computation on a real window, run live in CLIP | 3 · Teacher |
| h_t, u, three heads | real layer shapes and real head outputs | 4 · Student, sections 1–2 |
| loss = 0.806 | the same four terms on a real batch of 128 | 4 · "loss by hand" (matches `TotalLoss`) |
| τ = 0.301 from 6 validation windows | τ from thousands of validation windows | 5 · Evaluation A |
| timeline, min_len, gating | the same, with (α, k, min_len) chosen on validation | 5 · Evaluation B |
| event table | every real event, found or missed | 5 · Evaluation C |

The meaning of every metric is in [METRICS_GUIDE.md](METRICS_GUIDE.md). The full picture is in
`final_ie643_report/figures/pipeline_diagram.png`.
