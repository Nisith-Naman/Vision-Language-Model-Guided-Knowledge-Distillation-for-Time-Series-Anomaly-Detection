"""Audit the MetroPT-3 result: is 0.989 teacher AUPRC leakage, an easy test, or luck?

Checks (writes experiments/benchmark_v2/metropt/AUDIT.md):
  1. Partition overlap in time and identity (failures 1 / 2 / 3+4; normal blocks).
  2. Per-failure-event AUPRC of every model (test has only failures 3 and 4).
  3. Difficulty: AUPRC of single raw-sensor rules (window mean of one channel,
     direction fixed on TRAINING data, no fitting on test).
  4. Similarity: are test failure windows near-duplicates of training failure
     windows? (nearest-neighbour distance in standardised space vs. normal-normal)
  5. Uncertainty: block bootstrap over test sequences (failure events and normal
     blocks resampled as units) -> 95% interval for window AUPRC.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score as AP

from src.experiments.data import prepare_dataset

OUT = []


def say(s=""):
    print(s, flush=True)
    OUT.append(s)


def main():
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    data = prepare_dataset("metropt", cfg)
    tr, va, te = data.splits["train"], data.splits["val"], data.splits["test"]
    bench = ROOT / "experiments" / "benchmark_v2" / "metropt"
    with np.load(bench / "teacher_outputs.npz") as z:
        T = {k: z[k] for k in z.files}

    say("# MetroPT-3 audit: why is the teacher's test AUPRC 0.989?")
    say("")

    # 1. Overlap ---------------------------------------------------------------
    say("## 1. Partitions")
    for name, sp in (("train", tr), ("val", va), ("test", te)):
        ev = sorted({s["group_id"] for s in sp.sequences if s["labels"].any()})
        ts = [s["timestamps"] for s in sp.sequences if s.get("timestamps") is not None]
        rng = ""
        if ts:
            lo = min(pd.Timestamp(np.asarray(t)[0]) for t in ts)
            hi = max(pd.Timestamp(np.asarray(t)[-1]) for t in ts)
            rng = f", time span {lo:%Y-%m-%d} .. {hi:%Y-%m-%d}"
        say(f"- {name}: {len(sp.sequences)} sequences, {len(sp.X)} windows, "
            f"{int(sp.y_window.sum())} anomalous; failure sequences {ev}{rng}")
    man = json.loads((bench / "manifest.json").read_text())
    g = {k: set(man["partitions"][k]["groups"]) for k in ("train", "val", "test")}
    say(f"- shared sequence ids: train/val {len(g['train'] & g['val'])}, train/test {len(g['train'] & g['test'])}, "
        f"val/test {len(g['val'] & g['test'])}")
    say("")

    # Window -> sequence id for the test split
    seq_of = np.empty(len(te.X), dtype=object)
    for s in te.sequences:
        seq_of[s["window_indices"]] = s["group_id"]
    y = te.y_window.astype(int)
    failure_ids = sorted({sid for sid in set(seq_of) if y[seq_of == sid].any()})

    # 2. Per-event AUPRC -------------------------------------------------------
    say("## 2. Teacher and student AUPRC per test failure event (each event vs. all normal test windows)")
    m = pd.read_csv(bench / "metrics.csv", dtype={"seed": str})
    z = np.load(bench / "test_predictions_seed0.npz")
    say("| Model | " + " | ".join(failure_ids) + " | both |")
    say("|---|" + "---|" * (len(failure_ids) + 1))
    normal = y == 0
    for key in ("zeroshot", "probe", "coop", "coop_mv"):
        s = T[f"{key}_test"][:, 1]
        cells = [AP(y[(seq_of == f) | normal], s[(seq_of == f) | normal]) for f in failure_ids]
        say(f"| teacher {key} | " + " | ".join(f"{c:.3f}" for c in cells) + f" | {AP(y, s):.3f} |")
    say("")

    # 3. Single raw-sensor rules ---------------------------------------------
    say("## 3. How easy is the test? Single-sensor rules (direction fixed on training data)")
    say("Score = mean of one standardised channel over the window (or its negative, whichever ranks the "
        "*training* failure higher). Nothing is fitted on test.")
    say("")
    say("| Channel | train AUPRC | test AUPRC |")
    say("|---|---|---|")
    rows = []
    for f, name in enumerate(data.feature_names):
        tr_s, te_s = tr.X[:, :, f].mean(1), te.X[:, :, f].mean(1)
        sign = 1.0 if AP(tr.y_window, tr_s) >= AP(tr.y_window, -tr_s) else -1.0
        rows.append((name, AP(tr.y_window, sign * tr_s), AP(y, sign * te_s)))
    for name, a_tr, a_te in sorted(rows, key=lambda r: -r[2])[:6]:
        say(f"| {name} | {a_tr:.3f} | {a_te:.3f} |")
    best = max(rows, key=lambda r: r[2])
    say("")
    say(f"Best single sensor on test: **{best[0]}**, AUPRC {best[2]:.3f} (prevalence {y.mean():.3f}).")
    say("")

    # 4. Near-duplicate check -------------------------------------------------
    say("## 4. Are test failures near-duplicates of the training failure?")
    flat = lambda X: X.reshape(len(X), -1)
    tr_fail, tr_norm = flat(tr.X[tr.y_window == 1]), flat(tr.X[tr.y_window == 0])
    rng = np.random.default_rng(0)
    tr_norm = tr_norm[rng.choice(len(tr_norm), min(2000, len(tr_norm)), replace=False)]
    te_fail = flat(te.X[y == 1])
    te_norm = flat(te.X[y == 0])[rng.choice(int(normal.sum()), min(500, int(normal.sum())), replace=False)]

    def nn(a, b):
        d = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
        return np.sqrt(d.min(1)) / np.sqrt(a.shape[1])

    d_ff, d_fn, d_nn = nn(te_fail, tr_fail), nn(te_fail, tr_norm), nn(te_norm, tr_norm)
    say(f"- median NN distance (per value, std units): test failure -> train failure {np.median(d_ff):.3f}; "
        f"test failure -> train normal {np.median(d_fn):.3f}; test normal -> train normal {np.median(d_nn):.3f}")
    say(f"- exact duplicates (distance < 1e-6): {int((d_ff < 1e-6).sum())} of {len(d_ff)} test failure windows")
    say("")

    # 5. Block bootstrap ------------------------------------------------------
    say("## 5. Uncertainty: block bootstrap over test sequences (2000 resamples)")
    say("Failure events and normal blocks are resampled as whole units, since windows inside one event "
        "are strongly correlated.")
    say("")
    fail_units = [np.flatnonzero(seq_of == f) for f in failure_ids]
    norm_units = [np.flatnonzero(seq_of == sid) for sid in sorted(set(seq_of) - set(failure_ids))]
    st = {k: T[f"{k}_test"][:, 1] for k in ("probe", "coop", "coop_mv")}
    st["student gru_kd_coop (seed 0)"] = None
    stu = z["gru_kd_coop__score"] if "gru_kd_coop__score" in z.files else None
    say("| Model | AUPRC | 95% interval |")
    say("|---|---|---|")
    for key, s in st.items():
        if s is None:
            continue
        vals = []
        for _ in range(2000):
            fi = [fail_units[i] for i in rng.integers(len(fail_units), size=len(fail_units))]
            ni = [norm_units[i] for i in rng.integers(len(norm_units), size=len(norm_units))]
            idx = np.concatenate(fi + ni)
            vals.append(AP(y[idx], s[idx]))
        lo, hi = np.percentile(vals, [2.5, 97.5])
        say(f"| teacher {key} | {AP(y, s):.3f} | {lo:.3f} - {hi:.3f} |")
    say("")
    say(f"Number of independent failure events in test: {len(failure_ids)}. "
        "With two events the interval mainly reflects which normal blocks are drawn; "
        "it cannot capture failure-to-failure variation beyond these two events.")
    (bench / "AUDIT.md").write_text("\n".join(OUT), encoding="utf-8")
    print(f"\nwrote {bench / 'AUDIT.md'}")


if __name__ == "__main__":
    main()
