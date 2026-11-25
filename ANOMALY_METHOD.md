# Anomaly decision method

## What the system predicts

The frozen CLIP vision-language teacher sees rendered multivariate windows and
provides soft normal/anomaly targets during training. The GRU student learns
from those targets through temperature-scaled KL loss and from the dataset's
labels through window classification and per-timestep localization losses.
CLIP is not run during CSV inference; the deployed detector is the trained GRU
student.

For each row, the student returns a timestep anomaly score. A threshold chosen
on validation data turns that score into `is_student_anomaly`. The output also
keeps the score and student decision separate from optional contextual alerts.
This matters because a model probability answers “does this resemble the
anomaly class used for training?” It does not automatically mean “is this
amount unusually high for this account?”

## The three anomaly views

1. **Point anomaly** — an isolated student alert (or contextual alert for a
   one-row event).
2. **Contextual anomaly** — a feature differs sharply from the same entity's
   recent history. This check is available only when the uploaded CSV has an
   entity key such as cardholder, patient, machine, or well ID.
3. **Collective anomaly** — adjacent anomalous rows within one real time series
   form a sustained event. Independent transaction rows are kept as separate
   point events even when adjacent in the file.

For existing CSV consumers, `anomaly_type` retains `point_anomaly` and
`group_anomaly`; the new `anomaly_taxonomy` column maps a group event to
`collective_anomaly` and can also label `contextual_anomaly` explicitly.

When an entity ID is selected, the contextual check is causal: each row is
compared with up to the preceding 64 rows for that same ID. It uses each
feature's trailing median and median absolute deviation (MAD), with a robust
scale of `1.4826 × MAD` and a small scale floor for constant channels. The
largest per-feature robust z-score is reported as `contextual_robust_z`; the
feature responsible is reported as `contextual_driver_feature`. The first 16
rows for each ID establish history and are not context-scored. The starting
alert threshold is calibrated from normal validation rows at a 0.5% target
false-alarm rate, with a conservative minimum of 4.5 robust standard
deviations; it can be adjusted in the inference page and is saved with the
checkpoint.

The final `is_anomaly` flag is the union of the validation-thresholded student
decision and any enabled contextual alert. CSV columns retain both sources:
`is_student_anomaly`, `is_contextual_anomaly`, and `detection_basis`. If no
entity key is selected, contextual columns state that no baseline was
available; they do not silently compare different entities.

## Why CreditCard amounts can look surprising

The ULB CreditCard file is a pooled transaction table. Its public schema has
`Time`, anonymized PCA features `V1`–`V28`, `Amount`, and the fraud label
`Class`; it does not identify which cardholder made each row. The project
therefore scores the multivariate fraud pattern and treats rows as independent
transactions. A large `Amount` is not automatically fraud, and personal
spending baselines cannot be recovered from this file. To flag “unusually large
for this cardholder,” the input needs a real cardholder ID and repeated rows
per cardholder; select that ID in the contextual-check control.

On the current held-out CreditCard export, the student flagged 52 of 56,961
transactions: 48 matched fraud labels and 4 were false positives. That is
92.3% precision and 64.0% recall on 75 labeled frauds. The flagged transactions
had a median amount of $1.14; four unflagged legitimate transactions exceeded
$8,000, and the largest was $25,691.16. This verifies that this model is
recognizing the dataset's fraud labels from the multivariate feature pattern,
not simply treating high amounts as anomalies. The missed 27 frauds also show
that the detector is not perfect; the threshold expresses the precision/recall
tradeoff.

## Evaluation and limits

Thresholds are selected on validation labels, then fixed for test evaluation.
Report point precision, recall, F1, PR-AUC, ROC-AUC, and event-overlap metrics
as appropriate; do not use a point-adjusted metric that marks an entire event
correct after detecting only one timestep. Each dataset needs its own
validation threshold and a review of label meaning. Synthetic labels measure
recovery of the injected events, not discovery of every possible real-world
failure.

The contextual robust-z rule is deliberately small and interpretable. It needs
chronologically ordered rows and a valid entity key. Its validation-calibrated
threshold is a starting point, not a universal guarantee: check the observed
false-alarm rate on the target dataset before operational use. It is an
additional evidence channel, not a substitute for the labelled student
evaluation.

## References

- Sehili and Zhang, [*Multivariate Time Series Anomaly Detection: Fancy Algorithms and Flawed Evaluation Methodology*](https://arxiv.org/abs/2308.13068), 2023.
- ULB Machine Learning Group, [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
