"""Score a label-free CSV with the frozen-CLIP teacher and the distilled GRU student.

Without --answer-key it only reports what each model flags (the deployment
view). With the hidden key it also grades the predictions, per anomaly type.

    .venv\\Scripts\\python.exe scripts\\score_unlabeled.py --dataset metropt ^
        --csv data/unlabeled_tests/metropt_bootstrap_seed0.csv ^
        --answer-key data/unlabeled_tests/metropt_bootstrap_seed0.answer_key.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.evaluation.postprocess import event_scores, runs
from src.inference import dual


def score_frame(dataset: str, df: pd.DataFrame, with_teacher: bool = True, embedder=None) -> dict:
    """Return {model name: result dict} for the student, the labels-only GRU and the teacher."""
    bundle = dual.load_bundle(dataset)
    scaled = dual.scale_frame(df, bundle)
    results = {"student (distilled GRU)": dual.score_student(bundle, bundle.student_key, scaled)}
    if "gru_gt" in bundle.students:
        results["GRU labels-only"] = dual.score_student(bundle, "gru_gt", scaled)
    if with_teacher:
        results["teacher (frozen CLIP)"] = dual.score_teacher(bundle, scaled, embedder=embedder, max_windows=10000)
    return results


def grade(results: dict, key: pd.DataFrame) -> pd.DataFrame:
    y = key["label"].to_numpy().astype(int)
    kinds = key["anomaly_type"].fillna("").to_numpy()
    rows = []
    for name, r in results.items():
        ev = event_scores([y], [r["pred"]])
        row = {"model": name, "prevalence": y.mean(), "point AUPRC": average_precision_score(y, r["score"]),
               "flagged steps": int(r["pred"].sum()), "event precision": ev["event_precision"],
               "event recall": ev["event_recall"], "event F1": ev["event_f1"]}
        cum = np.r_[0, np.cumsum(r["pred"])]
        for t in sorted({k for k in kinds if k}):
            spans = [(s, e) for s, e in runs(y) if kinds[s] == t]
            row[f"found:{t}"] = f"{sum(cum[e] - cum[s] > 0 for s, e in spans)}/{len(spans)}"
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--answer-key", default=None)
    p.add_argument("--no-teacher", action="store_true", help="skip the (slow) CLIP teacher")
    p.add_argument("--out", default=None, help="write per-step scores/flags to this CSV")
    a = p.parse_args()
    df = pd.read_csv(a.csv)
    if "label" in df.columns:
        print("note: the CSV has a 'label' column; it is ignored for scoring")
    results = score_frame(a.dataset, df, with_teacher=not a.no_teacher)
    print(f"\n{len(df)} rows, model family '{a.dataset}'")
    for name, r in results.items():
        print(f"  {name:26s} flagged {int(r['pred'].sum()):6d} steps in {len(runs(r['pred'])):3d} intervals "
              f"({r['windows'] / max(r['seconds'], 1e-9):,.0f} windows/s)")
    if a.answer_key:
        g = grade(results, pd.read_csv(a.answer_key))
        print("\nGraded against the hidden answer key:")
        print(g.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    if a.out:
        out = df.copy()
        for name, r in results.items():
            tag = name.split(" ")[0].lower()
            out[f"{tag}_score"], out[f"{tag}_anomaly"] = r["score"], r["pred"]
        out.to_csv(a.out, index=False)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
