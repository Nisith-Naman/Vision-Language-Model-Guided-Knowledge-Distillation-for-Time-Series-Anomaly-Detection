"""Generalisation study: base vs. regularized training (EMA, augmentation, lr 1e-3, dropout 0.3).

For each dataset and model (labels-only GRU, proposed distilled student) the
configuration with the higher mean best-validation AUPRC over seeds is
selected. Test metrics are reported but never used for the choice. When the
regularized configuration wins, its best-validation-seed checkpoint is deployed
as checkpoints/v2_<dataset>_<model> (the previous one is kept as
checkpoints/v2base_<dataset>_<model>). Writes experiments/benchmark_v2/GENERALISATION.md.

    .venv\\Scripts\\python.exe scripts\\compare_generalisation.py [--deploy]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import average_precision_score as AP

from src.experiments.data import prepare_dataset
from src.training.checkpoint import load_checkpoint

BASE = ROOT / "experiments" / "benchmark_v2"
REG = ROOT / "experiments" / "benchmark_v2_regularized"
DATASETS = ["creditcard", "daphnet", "oil3w", "metropt"]
MODELS = {"gru_gt": "GRU, labels only", "gru_kd_coop": "GRU + KD (proposed)"}
PRETTY = {"creditcard": "Credit Card", "daphnet": "Daphnet FoG", "oil3w": "3W Oil", "metropt": "MetroPT-3"}


def val_scores(run: Path, ds: str, model: str) -> list[float]:
    hist = json.loads((run / ds / "training_histories.json").read_text())
    return [float(np.nanmax(h["val_auprc"])) for k, h in hist.items() if k.rsplit("_seed", 1)[0] == model]


def test_stats(run: Path, ds: str, model: str, level: str, col: str = "auprc"):
    m = pd.read_csv(run / ds / "metrics.csv", dtype={"seed": str})
    v = m[(m.model == model) & (m.level == level)][col].astype(float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def gap(prefix: str, ds: str, model: str, data) -> float:
    ck = ROOT / "checkpoints" / f"{prefix}_{ds}_{model}"
    net, _, _ = load_checkpoint(str(ck))
    net.eval()

    def score(X):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 512):
                out.append(net.predict_proba(torch.as_tensor(X[i:i + 512]))[0][:, 1])
        return np.concatenate(out)

    sp = data.splits
    return AP(sp["train"].y_window, score(sp["train"].X)) - AP(sp["test"].y_window, score(sp["test"].X))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--deploy", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(4)
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    lines = ["# Generalisation study: base vs. regularized training", "",
             "Regularized = weight EMA (0.99), gain/noise augmentation (0.15 / 0.05), learning rate 1e-3, dropout "
             "0.3, patience 6. Same splits, teachers, seeds and initial-weight seeds as the base runs. The choice "
             "per dataset and model uses **validation AUPRC only**; test numbers are reported afterwards.", "",
             "| Dataset | Model | Val AUPRC base → reg | Test window AUPRC base → reg | Test point AUPRC base → reg | "
             "Seed std (window) base → reg | Train−test gap base → reg | Selected |",
             "|---|---|---|---|---|---|---|---|"]
    decisions = {}
    for ds in DATASETS:
        if not (REG / ds / "metrics.csv").exists():
            continue
        pcfg = cfg["benchmark"]["per_dataset"].get(ds, {})
        data = prepare_dataset(ds, cfg, max_train_windows=int(pcfg.get("max_train_windows", 0)))
        for model, label in MODELS.items():
            vb, vr = np.mean(val_scores(BASE, ds, model)), np.mean(val_scores(REG, ds, model))
            wb, sb = test_stats(BASE, ds, model, "window")
            wr, sr = test_stats(REG, ds, model, "window")
            pb, _ = test_stats(BASE, ds, model, "point")
            pr, _ = test_stats(REG, ds, model, "point")
            gb = gap("v2" if not (ROOT / "checkpoints" / f"v2base_{ds}_{model}").exists() else "v2base", ds, model, data)
            gr = gap("v2reg", ds, model, data)
            choice = "regularized" if vr > vb + 1e-6 else "base"
            decisions[(ds, model)] = choice
            if abs(vr - vb) < 0.005:
                choice += f" (near-tie: val Δ {vr - vb:+.4f})"
            lines.append(f"| {PRETTY[ds]} | {label} | {vb:.3f} → {vr:.3f} | {wb:.3f} → {wr:.3f} | {pb:.3f} → {pr:.3f} | "
                         f"{sb:.3f} → {sr:.3f} | {gb:+.3f} → {gr:+.3f} | **{choice}** |")
        print(f"{ds} done", flush=True)
    lines += ["", "**Reading the selection.** Validation separates the two configurations only weakly: on MetroPT it "
              "is saturated (both ≈ 1.000; differences ≈ 0.0001, because validation holds a single failure) and on "
              "Daphnet and 3W the differences are below 0.005. These choices are therefore near-ties under a "
              "pre-set rule, not validated wins. What the regularized recipe does show consistently on test is lower "
              "seed variance and a smaller train−test gap where the base models overfit (Daphnet, MetroPT labels-only "
              "GRU), which is its purpose. Earlier, the same recipe lost to the base configuration on Daphnet when "
              "validation AUPRC was averaged over all five student variants (0.830 vs 0.834); that comparison fixed "
              "the configuration of the main ablation table.",
              "", "Deployment: the selected configuration's best-validation-seed checkpoint is served by the "
              "interface (`checkpoints/v2_<dataset>_<model>`). The main ablation tables in RESULTS.md use the base "
              "configuration for every variant, so all variants remain directly comparable."]
    if a.deploy:
        moved = []
        for (ds, model), choice in decisions.items():
            if choice != "regularized":
                continue
            cur, reg, bak = (ROOT / "checkpoints" / f"{x}_{ds}_{model}" for x in ("v2", "v2reg", "v2base"))
            if not bak.exists():
                shutil.copytree(cur, bak)
            shutil.rmtree(cur)
            shutil.copytree(reg, cur)
            moved.append(f"{ds}/{model}")
        lines += ["", f"Deployed regularized checkpoints: {', '.join(moved) if moved else 'none'} "
                  "(previous ones kept as `checkpoints/v2base_*`)."]
    (BASE / "GENERALISATION.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {BASE / 'GENERALISATION.md'}")


if __name__ == "__main__":
    main()
