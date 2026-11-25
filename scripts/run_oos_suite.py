"""Out-of-sample suite: generate label-free series for every dataset and grade them.

For each dataset and background mode (bootstrap / synthetic) and each seed, a
label-free CSV is generated (scripts/generate_unlabeled_test.py logic), scored
by the distilled student, the labels-only GRU and the frozen-CLIP teacher, and
graded against the hidden answer key. Writes
experiments/benchmark_v2/out_of_sample.csv and OUT_OF_SAMPLE.md.

    .venv\\Scripts\\python.exe scripts\\run_oos_suite.py
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

import pandas as pd
import yaml

from scripts.score_unlabeled import grade, score_frame
from src.data.unlabeled_generator import generate
from src.experiments.data import prepare_dataset
from src.teacher.clip_teacher import CLIPEmbedder

PRETTY = {"creditcard": "Credit Card", "daphnet": "Daphnet FoG", "oil3w": "3W Oil", "metropt": "MetroPT-3"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="creditcard,daphnet,oil3w,metropt")
    p.add_argument("--modes", default="bootstrap,synthetic")
    p.add_argument("--seeds", default="0,1")
    a = p.parse_args()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    out_dir = ROOT / "data" / "unlabeled_tests"
    out_dir.mkdir(parents=True, exist_ok=True)
    emb = CLIPEmbedder().load()
    frames = []
    for ds in a.datasets.split(","):
        pcfg = cfg["benchmark"]["per_dataset"].get(ds, {})
        data = prepare_dataset(ds, cfg, max_train_windows=int(pcfg.get("max_train_windows", 0)))
        for mode in a.modes.split(","):
            for seed in [int(s) for s in a.seeds.split(",")]:
                g = generate(data, length=36 * data.window_length, mode=mode, seed=seed)
                stem = out_dir / f"{ds}_{mode}_seed{seed}"
                g.frame.to_csv(f"{stem}.csv", index=False)
                g.answer_key.to_csv(f"{stem}.answer_key.csv", index=False)
                g.events.to_csv(f"{stem}.events.csv", index=False)
                if mode == "bootstrap" and seed == 0:
                    # Interface demo: the same label-free series with the hidden key as an optional column.
                    demo = g.frame.copy()
                    demo["label"] = g.answer_key["label"].to_numpy()
                    demo.to_csv(ROOT / "experiments" / "benchmark_v2" / ds / "demo_3_out_of_sample.csv", index=False)
                res = score_frame(ds, pd.read_csv(f"{stem}.csv"), with_teacher=True, embedder=emb)
                gr = grade(res, g.answer_key)
                gr.insert(0, "seed", seed)
                gr.insert(0, "mode", mode)
                gr.insert(0, "dataset", ds)
                frames.append(gr)
                print(gr.to_string(index=False, float_format=lambda v: f"{v:.3f}"), flush=True)
    df = pd.concat(frames, ignore_index=True)
    bench = ROOT / "experiments" / "benchmark_v2"
    df.to_csv(bench / "out_of_sample.csv", index=False)

    found_cols = [c for c in df.columns if c.startswith("found:")]
    lines = ["# Out-of-sample test on label-free generated series", "",
             "Each series is generated without labels; models see only the CSV. The answer key is used "
             "afterwards for grading. `bootstrap`: normal stretches of the held-out test split stitched together; "
             "`synthetic`: an AR(1) process fitted to normal training windows. Hidden anomalies: 3 real "
             "anomalies transplanted from the test split (`real_event`, the kind the models were trained on) "
             "and one each of six generic shapes they never saw. Numbers are averaged over seeds.", ""]
    for (ds, mode), g in df.groupby(["dataset", "mode"], sort=False):
        lines += [f"## {PRETTY.get(ds, ds)} — {mode}", "",
                  f"Prevalence {g['prevalence'].mean():.3f}.", "",
                  "| Model | Point AUPRC | Event precision | Event recall | Event F1 | " +
                  " | ".join(c.replace("found:", "") for c in found_cols if g[c].notna().any()) + " |",
                  "|---|---|---|---|---|" + "---|" * sum(g[c].notna().any() for c in found_cols)]
        for model, m in g.groupby("model", sort=False):
            found = []
            for c in found_cols:
                if not m[c].notna().any():
                    continue
                hit = sum(int(v.split("/")[0]) for v in m[c].dropna())
                tot = sum(int(v.split("/")[1]) for v in m[c].dropna())
                found.append(f"{hit}/{tot}")
            lines.append(f"| {model} | {m['point AUPRC'].mean():.3f} | {m['event precision'].mean():.3f} | "
                         f"{m['event recall'].mean():.3f} | {m['event F1'].mean():.3f} | " + " | ".join(found) + " |")
        lines.append("")
    (bench / "OUT_OF_SAMPLE.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {bench / 'OUT_OF_SAMPLE.md'}")


if __name__ == "__main__":
    main()
