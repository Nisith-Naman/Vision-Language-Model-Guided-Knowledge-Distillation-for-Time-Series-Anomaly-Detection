"""Choose between benchmark configurations for one dataset using validation data only.

Each candidate directory must contain ``<dataset>/training_histories.json`` from
``scripts/run_benchmark.py``. The score of a candidate is the mean, over every
student variant and seed, of the best validation AUPRC reached during training.
Test metrics are not read. The winning run is placed in the main benchmark
directory; the other run is archived next to it as ``<dataset>__<candidate>``.

    .venv\\Scripts\\python.exe scripts\\select_config.py --dataset daphnet ^
        --candidates experiments/benchmark_v2 experiments/benchmark_v2_daphnet_tuned
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def val_score(run_dir: Path) -> tuple[float, dict]:
    hist = json.loads((run_dir / "training_histories.json").read_text())
    per_variant: dict[str, list[float]] = {}
    for key, h in hist.items():
        per_variant.setdefault(key.rsplit("_seed", 1)[0], []).append(float(np.nanmax(h["val_auprc"])))
    means = {k: float(np.mean(v)) for k, v in per_variant.items()}
    return float(np.mean(list(means.values()))), means


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--candidates", nargs="+", required=True, help="first = main benchmark directory")
    args = p.parse_args()
    main_dir = ROOT / args.candidates[0]
    scores = {}
    for c in args.candidates:
        d = ROOT / c / args.dataset
        scores[c] = val_score(d)
        print(f"{c}: mean best validation AUPRC {scores[c][0]:.4f}  {scores[c][1]}")
    winner = max(scores, key=lambda c: scores[c][0])
    print(f"selected: {winner}")
    target = main_dir / args.dataset
    if winner != args.candidates[0]:
        archive = main_dir / f"{args.dataset}__base_config"
        if archive.exists():
            shutil.rmtree(archive)
        shutil.move(str(target), str(archive))
        shutil.copytree(ROOT / winner / args.dataset, target)
    record = {
        "dataset": args.dataset,
        "rule": "highest mean (over student variants and seeds) of best validation AUPRC; test not used",
        "scores": {c: {"mean": s[0], "per_variant": s[1]} for c, s in scores.items()},
        "selected": winner,
        "decided_at": datetime.now().isoformat(timespec="seconds"),
    }
    (target / "config_selection.json").write_text(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
