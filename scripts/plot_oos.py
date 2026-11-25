"""Plot teacher vs. student localization on a label-free out-of-sample series.

Uses the files written by scripts/run_oos_suite.py (bootstrap background,
seed 0). The models see only the CSV; the hidden answer key is drawn as the
grey "truth" track with each event's type.

    .venv\\Scripts\\python.exe scripts\\plot_oos.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.score_unlabeled import score_frame
from src.evaluation.postprocess import runs
from src.inference import dual
from src.teacher.clip_teacher import CLIPEmbedder

PRETTY = {"creditcard": "Credit Card", "daphnet": "Daphnet FoG", "oil3w": "3W Oil", "metropt": "MetroPT-3"}
TRACKS = [("student (distilled GRU)", "GRU + CLIP KD (student)", "#7F1D1D"),
          ("GRU labels-only", "GRU labels only", "#6366F1"),
          ("teacher (frozen CLIP)", "Teacher (frozen CLIP)", "#C2410C")]


def plot(dataset: str, mode: str, seed: int, embedder) -> Path:
    stem = ROOT / "data" / "unlabeled_tests" / f"{dataset}_{mode}_seed{seed}"
    df = pd.read_csv(f"{stem}.csv")
    key = pd.read_csv(f"{stem}.answer_key.csv")
    res = score_frame(dataset, df, with_teacher=True, embedder=embedder)
    bundle = dual.load_bundle(dataset)
    scaled = dual.scale_frame(df, bundle)
    y = key["label"].to_numpy()
    kinds = key["anomaly_type"].fillna("").to_numpy()
    t = np.arange(len(df))
    fig, ax = plt.subplots(3, 1, figsize=(12, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 2, 1.4]})
    for f in range(min(5, scaled.shape[1])):
        ax[0].plot(t, np.clip(scaled[:, f], -6, 6), lw=0.6, label=bundle.feature_names[f])
    for s, e in runs(y):
        for a in ax[:2]:
            a.axvspan(s, e, color="#94A3B8", alpha=0.3, lw=0)
        ax[0].text((s + e) / 2, 5.6, kinds[s].replace("_", " "), ha="center", fontsize=7, color="#334155")
    ax[0].set_ylabel("standardised"); ax[0].set_ylim(-6.5, 6.5)
    ax[0].legend(fontsize=7, ncol=5, loc="lower right")
    for row, (k, lab, color) in enumerate(TRACKS):
        r = res[k]
        ax[1].plot(t, r["score"] / max(r["threshold"], 1e-9), color=color, lw=1, label=lab)
        for s, e in runs(r["pred"]):
            ax[2].axvspan(s, e, ymin=(row + 1) / 4 + 0.02, ymax=(row + 2) / 4 - 0.02, color=color, lw=0)
    ax[1].axhline(1.0, color="black", ls=":", lw=1)
    ax[1].set_yscale("symlog", linthresh=1.0)
    ax[1].set_ylabel("score / threshold"); ax[1].legend(fontsize=7, loc="upper right")
    for s, e in runs(y):
        ax[2].axvspan(s, e, ymin=0.02, ymax=0.23, color="#94A3B8", lw=0)
    ax[2].set_yticks([0.125, 0.375, 0.625, 0.875], ["truth (hidden)", "student", "GRU labels", "teacher"])
    ax[2].set_xlabel("timestep (label-free generated series)")
    fig.suptitle(f"{PRETTY[dataset]}: out-of-sample, label-free series ({mode} background) — "
                 f"anomalies above the dotted line are flagged")
    fig.tight_layout()
    out = ROOT / "experiments" / "benchmark_v2" / "figures" / f"oos_{dataset}_{mode}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    emb = CLIPEmbedder().load()
    for ds in ("creditcard", "daphnet", "oil3w", "metropt"):
        print(plot(ds, "bootstrap", 0, emb))


if __name__ == "__main__":
    main()
