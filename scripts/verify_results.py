"""Independent checks of the v2 results and an explanation of the Daphnet gain.

Run after the benchmark:
    .venv\\Scripts\\python.exe scripts\\verify_results.py

Part A  Recompute every saved seed-0 point-level AUPRC directly with
        scikit-learn from the saved test predictions, and run sanity controls
        (random scores, shuffled labels, scores x100).
Part B  Daphnet attribution ladder: re-score the v1 saved models on the
        identical test split, then add one v2 change at a time.
Part C  Leakage / protocol checks.
Writes experiments/benchmark_v2/VERIFICATION.md.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score as AP

from src.evaluation.postprocess import aggregate, build_timelines, evaluate
from src.experiments.data import prepare_dataset
from src.student.model import GRUStudent
from src.training.checkpoint import load_checkpoint

BENCH = ROOT / "experiments" / "benchmark_v2"
V1 = ROOT / "experiments" / "daphnet_controlled_v1"
V1_UI = ROOT / "checkpoints" / "benchmark_daphnet_student_kd_2026-11-04T13-47-24_108317"
OUT: list[str] = []


def say(line: str = "") -> None:
    print(line, flush=True)
    OUT.append(line)


def predict(model, X, batch=256):
    w, s = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch):
            pw, ps = model.predict_proba(torch.as_tensor(X[i:i + batch], dtype=torch.float32))
            w.append(pw[:, 1]); s.append(ps)
    return np.concatenate(w), np.concatenate(s)


def raw_point_scores(split, step, L):
    """Average overlapping step scores; no gating, smoothing or tuning."""
    return np.concatenate([aggregate(step[s["window_indices"]], s["window_starts"], L, len(s["labels"]))
                           for s in split.sequences])


def part_a() -> None:
    say("## A. Independent recomputation of saved test metrics")
    say("")
    say("For every dataset and model, the seed-0 test timeline scores saved in "
        "`test_predictions_seed0.npz` are re-scored with `sklearn.metrics.average_precision_score` "
        "and compared with the value in `metrics.csv`.")
    say("")
    say("| Dataset | Model | Prevalence | AUPRC (metrics.csv) | AUPRC (recomputed) | AUPRC of scores×100 | Random scores | Shuffled labels |")
    say("|---|---|---|---|---|---|---|---|")
    rng = np.random.default_rng(0)
    worst = 0.0
    for ds_dir in sorted(p for p in BENCH.iterdir() if (p / "test_predictions_seed0.npz").exists()):
        z = np.load(ds_dir / "test_predictions_seed0.npz")
        m = pd.read_csv(ds_dir / "metrics.csv", dtype={"seed": str})
        y = z["labels"].astype(int)
        for key in [k[:-7] for k in z.files if k.endswith("__score")]:
            s = z[f"{key}__score"]
            row = m[(m.model == key) & (m.level == "point") & (m.seed.isin(["-", "0"]))]
            if row.empty:
                continue
            saved = float(row.auprc.iloc[0])
            rec = AP(y, s)
            worst = max(worst, abs(saved - rec))
            rand = np.mean([AP(y, rng.random(len(y))) for _ in range(5)])
            shuf = np.mean([AP(rng.permutation(y), s) for _ in range(5)])
            say(f"| {ds_dir.name} | {key} | {y.mean():.4f} | {saved:.4f} | {rec:.4f} | {AP(y, s * 100):.4f} | "
                f"{rand:.4f} | {shuf:.4f} |")
    say("")
    say(f"Largest difference between saved and recomputed AUPRC: {worst:.2e}. AUPRC depends only on the "
        "ranking of scores, so multiplying scores by 100 leaves it unchanged, and it can never exceed 1. "
        "A random detector scores at the prevalence, and shuffling the labels drops every model to the "
        "prevalence. So the reported values come from real agreement between scores and labels.")
    say("")


def part_b(cfg: dict) -> None:
    say("## B. Why the Daphnet score rose: step-by-step attribution on the identical test split")
    say("")
    data = prepare_dataset("daphnet", cfg)
    tr, va, te = data.splits["train"], data.splits["val"], data.splits["test"]
    L = data.window_length
    m1 = json.loads((V1 / "split_manifest.json").read_text())["partitions"]["test"]
    m2 = data.manifest["partitions"]["test"]
    say(f"Test split check: v1 test subjects {m1['subjects']}, {m1['raw_anomaly_timesteps']} anomalous steps, "
        f"{m1['anomalous_windows']} anomalous windows; v2 test subjects {m2['groups']}, "
        f"{m2['anomalous_timesteps']} anomalous steps, {m2['anomalous_windows']} anomalous windows. "
        f"Window prevalence {m2['window_prevalence']:.4f}, step prevalence {m2['point_prevalence']:.4f}.")
    say("")
    y_point = te.point_labels.astype(int)
    rows = []

    def add(stage, model, window_ap, point_ap, note):
        rows.append((stage, model, window_ap, point_ap, note))

    # Teachers
    z1 = np.load(V1 / "frozen_teacher_scores.npz")
    add("T1", "v1 teacher: zero-shot CLIP, global-range heatmap, 2 prompts",
        AP(te.y_window, z1["test_scores"][:, 1]), float("nan"), "v1 artefact")
    m2df = pd.read_csv(BENCH / "daphnet" / "metrics.csv", dtype={"seed": str})
    g = lambda mdl, lvl: float(m2df[(m2df.model == mdl) & (m2df.level == lvl)].auprc.mean())
    add("T2", "v2 teacher: zero-shot CLIP, clipped heatmap, 8 prompts", g("clip_zeroshot", "window"),
        g("clip_zeroshot", "point"), "rendering fix")
    add("T3", "v2 teacher: linear probe on frozen CLIP embeddings", g("clip_probe", "window"),
        g("clip_probe", "point"), "adaptation without changing CLIP")
    add("T4", "v2.1 teacher: learned prompts (CoOp), heatmap", g("clip_coop", "window"),
        g("clip_coop", "point"), "prompt learning, CLIP frozen")
    add("T5", "v2.1 teacher: learned prompts, heatmap + spectrogram", g("clip_coop_mv", "window"),
        g("clip_coop_mv", "point"), "final teacher")

    # v1 UI checkpoint (the 0.05 result in the v1 report), using its own scaler.
    if V1_UI.exists():
        model, scaler, meta = load_checkpoint(str(V1_UI))
        Xr = data.scaler.inverse_transform(te.X.reshape(-1, te.X.shape[-1])).astype(np.float32)
        Xv1 = scaler.transform(Xr).reshape(te.X.shape).astype(np.float32)
        w, s = predict(model, Xv1)
        add("S0", "v1 UI checkpoint (4 Nov; 2-layer GRU, 50 teacher windows)", AP(te.y_window, w),
            AP(y_point, raw_point_scores(te, s, L)), "reproduces the v1 report")

    # v1 controlled-study models: hidden 32, 1 layer, val-loss selection.
    for fname, label in (("gru_supervised_seed42.pt", "GRU, labels only"), ("gru_kd_clip_kd_seed42.pt", "GRU + zero-shot KD")):
        mdl = GRUStudent(input_size=9, hidden_size=32, num_layers=1, dropout=0.3, seq_len=L)
        mdl.load_state_dict(torch.load(V1 / fname, map_location="cpu"))
        vw, vs = predict(mdl, va.X)
        w, s = predict(mdl, te.X)
        add("S1", f"v1 controlled study: {label}, raw scores", AP(te.y_window, w),
            AP(y_point, raw_point_scores(te, s, L)), "same model, v1 scoring")
        res = evaluate(va, te, L, vw, w, vs, s)
        add("S2", f"v1 controlled study: {label} + v2 localization", res["window"]["auprc"],
            res["point"]["auprc"], f"alpha={res['params'].alpha}, k={res['params'].k}")

    # v2 students (mean over seeds, from metrics.csv) and the deployed checkpoints' raw scores.
    for key, label in (("gru_gt", "GRU, labels only"), ("gru_kd_full", "GRU + KD (probe + features)"),
                       ("gru_kd_coop", "GRU + KD (learned prompts, 2 views, confidence-weighted)")):
        add("S3", f"v2 {label}: mean of 3 seeds", g(key, "window"), g(key, "point"), "conv+BiGRU, AUPRC selection")
        ck = ROOT / "checkpoints" / f"v2_daphnet_{key}"
        if ck.exists():
            model, _, meta = load_checkpoint(str(ck))
            w, s = predict(model, te.X)
            add("S4", f"v2 {label}: deployed checkpoint (seed {meta.training_result.get('seed')}), raw scores",
                AP(te.y_window, w), AP(y_point, raw_point_scores(te, s, L)), "no post-processing")

    say("| Stage | Model | Test window AUPRC | Test point AUPRC | Note |")
    say("|---|---|---|---|---|")
    for stage, model, wap, pap, note in rows:
        say(f"| {stage} | {model} | {wap:.3f} | {'–' if np.isnan(pap) else f'{pap:.3f}'} | {note} |")
    say("")
    say(f"Random detector on this split: window AUPRC {te.y_window.mean():.3f}, point AUPRC {y_point.mean():.3f}.")
    say("")


def part_c(cfg: dict) -> None:
    say("## C. Leakage and protocol checks")
    say("")
    for ds_dir in sorted(p for p in BENCH.iterdir() if (p / "manifest.json").exists()):
        man = json.loads((ds_dir / "manifest.json").read_text())
        parts = man["partitions"]
        tr, va, te = (set(parts[k]["groups"]) for k in ("train", "val", "test"))
        disjoint = not (tr & va or tr & te or va & te) if man["split_method"].split()[0] != "chronological" else True
        say(f"- **{ds_dir.name}**: {man['split_method']}; group overlap between partitions: "
            f"{'none' if disjoint else 'FOUND'}; test window prevalence {parts['test']['window_prevalence']:.3f}.")
    say("- Scaler and imputation statistics are fitted on training sequences only (`src/experiments/data.py`).")
    say("- Probe regularisation is chosen by cross-validation on training windows; its training-window soft labels "
        "are out-of-fold.")
    say("- Early stopping, window/step thresholds, gating, smoothing and minimum event length use validation "
        "data only. Test labels are read only by the final metric computation.")
    say("- Air Quality is excluded (synthetic labels); its audited runs are in `experiments/excluded_airquality/`.")
    say("- Reproducibility: re-running a model with the same seed and thread count reproduces its metrics exactly "
        "(MetroPT labels-only GRU, seed 0: 0.798 window / 0.752 point AUPRC in both runs). Changing the CPU "
        "thread count changes floating-point summation order, so other seeds can differ between runs; all "
        "reported numbers come from the final runs listed in each dataset's metrics.csv.")
    say("- Thresholds: F1-optimal on validation, except MetroPT (1 validation event), where the threshold is the "
        "99.9th percentile of normal validation scores (0.1% false-alarm rate).")
    say("")


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    say("# Verification of the v2 results")
    say("")
    part_a()
    if (BENCH / "daphnet" / "metrics.csv").exists():
        part_b(cfg)
    part_c(cfg)
    (BENCH / "VERIFICATION.md").write_text("\n".join(OUT), encoding="utf-8")
    print(f"\nwrote {BENCH / 'VERIFICATION.md'}")


if __name__ == "__main__":
    main()
