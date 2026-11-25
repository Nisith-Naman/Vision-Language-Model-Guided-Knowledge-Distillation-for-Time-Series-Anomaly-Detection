# Out-of-sample test on label-free generated series

Each series is generated without labels; models see only the CSV. The answer key is used afterwards for grading. `bootstrap`: normal stretches of the held-out test split stitched together; `synthetic`: an AR(1) process fitted to normal training windows. Hidden anomalies: 3 real anomalies transplanted from the test split (`real_event`, the kind the models were trained on) and one each of six generic shapes they never saw. Numbers are averaged over seeds.

## Credit Card — bootstrap

Prevalence 0.141.

| Model | Point AUPRC | Event precision | Event recall | Event F1 | drift | flatline | level_shift | noise_burst | oscillation | real_event | spike |
|---|---|---|---|---|---|---|---|---|---|---|---|
| student (distilled GRU) | 0.426 | 1.000 | 0.333 | 0.490 | 1/2 | 0/2 | 1/2 | 0/2 | 0/2 | 4/6 | 0/2 |
| GRU labels-only | 0.400 | 0.971 | 0.389 | 0.531 | 1/2 | 0/2 | 1/2 | 1/2 | 0/2 | 4/6 | 0/2 |
| teacher (frozen CLIP) | 0.205 | 0.500 | 0.056 | 0.100 | 0/2 | 0/2 | 0/2 | 1/2 | 0/2 | 0/6 | 0/2 |

## Credit Card — synthetic

Prevalence 0.117.

| Model | Point AUPRC | Event precision | Event recall | Event F1 | drift | flatline | level_shift | noise_burst | oscillation | real_event | spike |
|---|---|---|---|---|---|---|---|---|---|---|---|
| student (distilled GRU) | 0.225 | 1.000 | 0.222 | 0.350 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 4/6 | 0/2 |
| GRU labels-only | 0.337 | 1.000 | 0.333 | 0.490 | 0/2 | 0/2 | 1/2 | 0/2 | 0/2 | 5/6 | 0/2 |
| teacher (frozen CLIP) | 0.323 | 0.000 | 0.000 | 0.000 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/6 | 0/2 |

## Daphnet FoG — bootstrap

Prevalence 0.175.

| Model | Point AUPRC | Event precision | Event recall | Event F1 | drift | flatline | level_shift | noise_burst | oscillation | real_event | spike |
|---|---|---|---|---|---|---|---|---|---|---|---|
| student (distilled GRU) | 0.328 | 0.432 | 0.611 | 0.505 | 2/2 | 1/2 | 0/2 | 1/2 | 2/2 | 5/6 | 0/2 |
| GRU labels-only | 0.307 | 0.478 | 0.611 | 0.536 | 2/2 | 0/2 | 1/2 | 1/2 | 2/2 | 5/6 | 0/2 |
| teacher (frozen CLIP) | 0.239 | 0.536 | 0.500 | 0.517 | 1/2 | 0/2 | 0/2 | 1/2 | 2/2 | 5/6 | 0/2 |

## Daphnet FoG — synthetic

Prevalence 0.169.

| Model | Point AUPRC | Event precision | Event recall | Event F1 | drift | flatline | level_shift | noise_burst | oscillation | real_event | spike |
|---|---|---|---|---|---|---|---|---|---|---|---|
| student (distilled GRU) | 0.405 | 1.000 | 0.222 | 0.364 | 0/2 | 0/2 | 0/2 | 0/2 | 1/2 | 3/6 | 0/2 |
| GRU labels-only | 0.426 | 1.000 | 0.167 | 0.282 | 0/2 | 0/2 | 0/2 | 0/2 | 1/2 | 2/6 | 0/2 |
| teacher (frozen CLIP) | 0.367 | 0.000 | 0.000 | 0.000 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/6 | 0/2 |

## 3W Oil — bootstrap

Prevalence 0.295.

| Model | Point AUPRC | Event precision | Event recall | Event F1 | drift | flatline | level_shift | noise_burst | oscillation | real_event | spike |
|---|---|---|---|---|---|---|---|---|---|---|---|
| student (distilled GRU) | 0.712 | 0.833 | 0.556 | 0.660 | 1/2 | 0/2 | 2/2 | 0/2 | 1/2 | 5/6 | 1/2 |
| GRU labels-only | 0.695 | 0.817 | 0.667 | 0.730 | 1/2 | 0/2 | 2/2 | 1/2 | 1/2 | 6/6 | 1/2 |
| teacher (frozen CLIP) | 0.623 | 0.675 | 0.444 | 0.533 | 1/2 | 0/2 | 1/2 | 0/2 | 0/2 | 5/6 | 1/2 |

## 3W Oil — synthetic

Prevalence 0.280.

| Model | Point AUPRC | Event precision | Event recall | Event F1 | drift | flatline | level_shift | noise_burst | oscillation | real_event | spike |
|---|---|---|---|---|---|---|---|---|---|---|---|
| student (distilled GRU) | 0.687 | 1.000 | 0.944 | 0.971 | 2/2 | 2/2 | 2/2 | 1/2 | 2/2 | 6/6 | 2/2 |
| GRU labels-only | 0.720 | 1.000 | 1.000 | 1.000 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 6/6 | 2/2 |
| teacher (frozen CLIP) | 0.187 | 1.000 | 1.000 | 1.000 | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 6/6 | 2/2 |

## MetroPT-3 — bootstrap

Prevalence 0.301.

| Model | Point AUPRC | Event precision | Event recall | Event F1 | drift | flatline | level_shift | noise_burst | oscillation | real_event | spike |
|---|---|---|---|---|---|---|---|---|---|---|---|
| student (distilled GRU) | 0.687 | 0.833 | 0.278 | 0.417 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 5/6 | 0/2 |
| GRU labels-only | 0.453 | 0.000 | 0.000 | 0.000 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/6 | 0/2 |
| teacher (frozen CLIP) | 0.737 | 1.000 | 0.333 | 0.500 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 6/6 | 0/2 |

## MetroPT-3 — synthetic

Prevalence 0.292.

| Model | Point AUPRC | Event precision | Event recall | Event F1 | drift | flatline | level_shift | noise_burst | oscillation | real_event | spike |
|---|---|---|---|---|---|---|---|---|---|---|---|
| student (distilled GRU) | 0.754 | 0.917 | 0.389 | 0.515 | 0/2 | 1/2 | 0/2 | 1/2 | 0/2 | 5/6 | 0/2 |
| GRU labels-only | 0.424 | 0.000 | 0.000 | 0.000 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 0/6 | 0/2 |
| teacher (frozen CLIP) | 0.799 | 1.000 | 0.278 | 0.432 | 0/2 | 0/2 | 0/2 | 0/2 | 0/2 | 5/6 | 0/2 |
