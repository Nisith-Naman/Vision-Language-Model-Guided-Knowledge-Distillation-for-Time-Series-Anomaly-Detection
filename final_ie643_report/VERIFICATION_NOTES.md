# Verification notes

Prepared 4 November 2026 from the project copy at `C:\Users\ASUS\Downloads\IE643 primary assessment - Copy` and the locally downloaded IE643 template bundle. This ledger records the evidence behind report claims and items that still require student confirmation.

## Evidence ledger

| Claim | Evidence inspected | Status |
|---|---|---|
| Official layout | `IE643_Project_Report.tex` and PDF in the supplied template folder; margins, header, and required section order copied into `main.tex` | Verified against the downloaded template. Google Drive itself was inaccessible through the browser tool, but the downloaded bundle was present locally. |
| Project architecture | `src/data/`, `src/teacher/`, `src/student/`, `src/training/`, `src/evaluation/`, `src/inference/`, `app.py`, `config.yaml` | Verified. CLIP is frozen and scores heatmap windows; GRU student predicts a window class and timestep scores. CLIP is not fine-tuned. |
| Distillation loss | `src/training/trainer.py`, actual saved `run_config.json` files | Verified: weighted localization BCE + window CE + `lambda_kd * T^2 * KL`, with `T=2`, `beta=1`, `lambda_kd=0.5`, positive weight 10 for saved KD runs. Validation omits teacher targets, so validation loss excludes KD. |
| Dataset loaders and shapes | `src/data/datasets/*.py`, `config.yaml`, local dataset files, `DATA_PIPELINE.md` | Verified for preprocessing policy, feature choices, labels, and window sizes. Local Daphnet count is 17 files, 10 subjects, 35 valid segments, 1,140,835 retained steps. Local 3W copy contains 1,984 instances and approximately 50.9M rows. |
| Air Quality evaluation | `checkpoints/benchmark_airquality_student_kd_2026-11-02T20-01-08_203342/evaluation_metrics.csv`, `metadata.json` | Verified saved held-out metrics; anomalies are semi-synthetic injections, not naturally observed ground truth. |
| Credit Card evaluation | `checkpoints/benchmark_creditcard_student_kd_2026-11-03T09-42-51_355111/evaluation_metrics.csv`, `metadata.json` | Verified saved held-out metrics at threshold 0.5. Dataset has no cardholder identity; do not interpret sequences as one person's spending history. |
| Daphnet KD evaluation | `checkpoints/benchmark_daphnet_student_kd_2026-11-04T13-47-24_108317/evaluation_metrics.csv`, `metadata.json` | Verified. Step precision 0.023, recall 0.999, F1 0.045; 220,985 false-positive points. Teacher-scored training subset contains 50 windows. |
| Daphnet baseline | `checkpoints/benchmark_daphnet_student_baseline_2026-11-04T10-40-45_515763/evaluation_metrics.csv`, `metadata.json` | Verified separate saved run. Test support differs from KD run; the report does not claim a paired gain or loss. |
| 3W and MetroPT completion | `checkpoints/benchmark_oil3w_student_kd_2026-11-02T23-14-48_134855/` and checkpoint directory listing | 3W has training history/checkpoint but no `evaluation_metrics.csv`. No MetroPT checkpoint or final evaluation artifact was found. |
| Test count | `.venv` project test suite run on 4 November 2026 | 89 passed. Existing documentation saying 79 is stale and is not repeated as current. |
| Runtime/hardware | Runtime observed while preparing this report | Windows 11, Ryzen 5 7535HS, 15.3 GB RAM, Python 3.14.4, PyTorch 2.15.0+cpu; CUDA unavailable. Checkpoints do not store the training host, so these details describe the observed report-preparation environment only. |
| GitHub link | Repository root and `.git`/remote inspection | No Git metadata or remote URL is present. Report includes an explicit team-input placeholder. |
| Review boundary | Repository documentation and file timestamps | No prep-presentation date, notes, or commit history found. The report keeps both required subheadings but does not assign dated runs to a review phase without evidence. |

## Discrepancies and limitations

- Existing `report/` prediction exports do not reproduce the saved checkpoint metrics. Credit Card export shows a different confusion count, while Daphnet export contains no flagged points despite the latest evaluation file recording many. Their model/threshold provenance is unclear; they are excluded from quantitative report claims.
- `DATA_PIPELINE.md` describes a 79-test suite; the executed current suite reported 89 passing tests.
- The latest Daphnet baseline and KD evaluation files have different test class counts and supports. They are not a controlled ablation.
- Air Quality scores test recovery of injected anomalies. They do not measure naturally occurring anomaly detection.
- The Credit Card data are a pooled transaction sequence with no cardholder ID. It cannot establish a personal spending baseline or support a per-person contextual claim.
- MetroPT code contains preprocessing and event labels, but no saved trained checkpoint or final evaluation was found. 3W training is not accompanied by final test metrics.
- CLIP has no teacher-only saved benchmark in the inspected artifacts. Its prompt strings are configurable and the checkpoint does not persist the exact prompts used for every run.
- The recorded evaluation artifacts are single runs; confidence intervals and repeated-seed comparisons are absent.
- The built-in LaTeX editor successfully compiled the final source after its initial TeX package cache was populated. Its compile tool does not export a PDF or return a page count. The source compiles; exact page count and page-by-page visual inspection remain to be confirmed in the editor's live preview.

## Student input required before final submission

1. Replace `[TEAM NAME -- replace before submission]` and `[MEMBERS AND ROLL NUMBERS -- replace before submission]` in `main.tex`.
2. Replace `[ADD TEAM GITHUB URL]` with the actual repository link or remove it after the repository is created.
3. Add the prep-presentation review date and use project history to assign work confidently to before/after phases.
4. Confirm that the result checkpoints and dataset copies are the team's final approved versions.

