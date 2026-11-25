# MetroPT-3 audit: why is the teacher's test AUPRC 0.989?

## 1. Partitions
- train: 159 sequences, 6597 windows, 45 anomalous; failure sequences ['metropt-event-1']
- val: 35 sequences, 1251 windows, 11 anomalous; failure sequences ['metropt-event-2'], time span 2020-05-29 .. 2020-07-31
- test: 42 sequences, 1685 windows, 104 anomalous; failure sequences ['metropt-event-3', 'metropt-event-4'], time span 2020-06-03 .. 2020-09-01
- shared sequence ids: train/val 0, train/test 0, val/test 0

## 2. Teacher and student AUPRC per test failure event (each event vs. all normal test windows)
| Model | metropt-event-3 | metropt-event-4 | both |
|---|---|---|---|
| teacher zeroshot | 0.035 | 0.003 | 0.037 |
| teacher probe | 0.946 | 0.089 | 0.931 |
| teacher coop | 1.000 | 0.498 | 0.989 |
| teacher coop_mv | 0.998 | 1.000 | 0.998 |

## 3. How easy is the test? Single-sensor rules (direction fixed on training data)
Score = mean of one standardised channel over the window (or its negative, whichever ranks the *training* failure higher). Nothing is fitted on test.

| Channel | train AUPRC | test AUPRC |
|---|---|---|
| DV_pressure | 0.239 | 0.917 |
| Motor_current | 0.098 | 0.904 |
| Oil_temperature | 0.156 | 0.865 |
| TP2 | 0.114 | 0.845 |
| Towers | 0.061 | 0.745 |
| H1 | 0.041 | 0.676 |

Best single sensor on test: **DV_pressure**, AUPRC 0.917 (prevalence 0.062).

## 4. Are test failures near-duplicates of the training failure?
- median NN distance (per value, std units): test failure -> train failure 0.510; test failure -> train normal 0.421; test normal -> train normal 0.673
- exact duplicates (distance < 1e-6): 0 of 104 test failure windows

## 5. Uncertainty: block bootstrap over test sequences (2000 resamples)
Failure events and normal blocks are resampled as whole units, since windows inside one event are strongly correlated.

| Model | AUPRC | 95% interval |
|---|---|---|
| teacher probe | 0.931 | 0.089 - 1.000 |
| teacher coop | 0.989 | 0.498 - 1.000 |
| teacher coop_mv | 0.998 | 0.995 - 1.000 |

Number of independent failure events in test: 2. With two events the interval mainly reflects which normal blocks are drawn; it cannot capture failure-to-failure variation beyond these two events.