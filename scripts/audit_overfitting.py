"""Overfitting audit for the final models on every dataset.

For the deployed students (labels-only GRU and distilled student, best
validation seed) and for the teachers, report window AUPRC on train /
validation / test next to the split's prevalence. Teacher train scores are
out-of-fold (cross-fitted), which is how they are used for distillation.
From the training histories, report the best epoch and how much validation
AUPRC fell between the best epoch and the last epoch trained (the overfitting
that early stopping prevented). Writes experiments/<bench>/OVERFITTING.md.

    .venv\\Scripts\\python.exe scripts\\audit_overfitting.py [--bench experiments/benchmark_v2] [--prefix v2]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score as AP

from src.experiments.data import prepare_dataset
from src.training.checkpoint import load_checkpoint

DATASETS = ["creditcard", "daphnet", "oil3w", "metropt"]
PRETTY = {"creditcard": "Credit Card", "daphnet": "Daphnet FoG", "oil3w": "3W Oil", "metropt": "MetroPT-3"}


def window_scores(model, X, batch=512):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(model.predict_proba(torch.as_tensor(X[i:i + batch]))[0][:, 1])
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", default="experiments/benchmark_v2")
    p.add_argument("--prefix", default="v2")
    a = p.parse_args()
    torch.set_num_threads(2)
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    bench = ROOT / a.bench
    lines = ["# Overfitting audit", "",
             "Window AUPRC per split (prevalence in brackets). A large train-minus-test gap with a low test score "
             "means overfitting; a high test score with a small gap means the model generalises. Teacher train "
             "scores are out-of-fold. Students: deployed checkpoint (best validation seed).", ""]
    for ds in DATASETS:
        if not (bench / ds / "metrics.csv").exists():
            continue
        pcfg = cfg["benchmark"]["per_dataset"].get(ds, {})
        data = prepare_dataset(ds, cfg, max_train_windows=int(pcfg.get("max_train_windows", 0)))
        sp = data.splits
        prev = {k: float(sp[k].y_window.mean()) for k in sp}
        lines += [f"## {PRETTY[ds]}", "",
                  f"Prevalence: train {prev['train']:.3f}, val {prev['val']:.3f}, test {prev['test']:.3f}.", "",
                  "| Model | Train | Val | Test | Train − Test |", "|---|---|---|---|---|"]
        for key, label in (("gru_gt", "GRU, labels only"), ("gru_kd_coop", "GRU + KD (proposed)")):
            ck = ROOT / "checkpoints" / f"{a.prefix}_{ds}_{key}"
            if not ck.exists():
                continue
            model, _, meta = load_checkpoint(str(ck))
            model.eval()
            r = {k: AP(sp[k].y_window, window_scores(model, sp[k].X)) for k in ("train", "val", "test")}
            lines.append(f"| {label} (seed {meta.training_result.get('seed')}) | {r['train']:.3f} | {r['val']:.3f} | "
                         f"{r['test']:.3f} | {r['train'] - r['test']:+.3f} |")
        tz = bench / ds / "teacher_outputs.npz"
        if tz.exists():
            z = np.load(tz)
            tr_meta = json.loads((bench / ds / "teacher_outputs.json").read_text())["identity"]
            n = len(sp["train"].X)
            idx = (np.arange(n) if n <= tr_meta["n_train"]
                   else np.linspace(0, n - 1, tr_meta["n_train"]).astype(int))
            ytr = sp["train"].y_window[idx]
            for key, label in (("probe", "Teacher: linear probe"), ("coop_mv", "Teacher: learned prompts, 2 views")):
                r = (AP(ytr, z[f"{key}_train"][:, 1]), AP(sp["val"].y_window, z[f"{key}_val"][:, 1]),
                     AP(sp["test"].y_window, z[f"{key}_test"][:, 1]))
                lines.append(f"| {label} (train = out-of-fold) | {r[0]:.3f} | {r[1]:.3f} | {r[2]:.3f} | "
                             f"{r[0] - r[2]:+.3f} |")
        hist = json.loads((bench / ds / "training_histories.json").read_text())
        lines += ["", "Training dynamics of the base-configuration runs (all seeds): best epoch, and validation AUPRC "
                  "lost between the best and the last epoch trained (protected by early stopping):", ""]
        for key in ("gru_gt", "gru_kd_coop"):
            hs = [h for k, h in hist.items() if k.rsplit("_seed", 1)[0] == key]
            if not hs:
                continue
            be = [h["best_epoch"] for h in hs]
            drop = [float(np.nanmax(h["val_auprc"]) - h["val_auprc"][-1]) for h in hs]
            lines.append(f"- {key}: best epochs {be}; val-AUPRC drop after best {np.round(drop, 3).tolist()}")
        lines.append("")
        print(f"{ds} done", flush=True)
    (bench / "OVERFITTING.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {bench / 'OVERFITTING.md'}")


if __name__ == "__main__":
    main()
