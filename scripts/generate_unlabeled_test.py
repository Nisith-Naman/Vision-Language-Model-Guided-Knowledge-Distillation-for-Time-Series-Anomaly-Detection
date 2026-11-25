"""Write a label-free out-of-sample test CSV (+ hidden answer key) for a trained dataset.

    .venv\\Scripts\\python.exe scripts\\generate_unlabeled_test.py --dataset metropt
    .venv\\Scripts\\python.exe scripts\\generate_unlabeled_test.py --dataset daphnet --mode synthetic --length 6000

Outputs in data/unlabeled_tests/:
    <dataset>_<mode>_seed<k>.csv              feature columns + timestamp, NO labels (upload this)
    <dataset>_<mode>_seed<k>.answer_key.csv   per-step label and anomaly type (keep hidden)
    <dataset>_<mode>_seed<k>.events.csv       one row per injected event
Score it with scripts/score_unlabeled.py or upload the CSV on page 8.
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

import yaml

from src.data.unlabeled_generator import generate
from src.experiments.data import prepare_dataset


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=["creditcard", "daphnet", "oil3w", "metropt"])
    p.add_argument("--mode", default="bootstrap", choices=["bootstrap", "synthetic"])
    p.add_argument("--length", type=int, default=None, help="timesteps (default 36 windows)")
    p.add_argument("--events", type=int, default=9,
                   help="default 9: 3 transplanted real anomalies + one of each generic shape")
    p.add_argument("--types", default=None,
                   help="comma list from real_event,spike,level_shift,drift,noise_burst,oscillation,flatline")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="data/unlabeled_tests")
    a = p.parse_args()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    pcfg = cfg["benchmark"]["per_dataset"].get(a.dataset, {})
    data = prepare_dataset(a.dataset, cfg, max_train_windows=int(pcfg.get("max_train_windows", 0)))
    length = a.length or 36 * data.window_length
    kwargs = {"types": tuple(a.types.split(","))} if a.types else {}
    g = generate(data, length=length, n_events=a.events, mode=a.mode, seed=a.seed, **kwargs)
    out = ROOT / a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"{a.dataset}_{a.mode}_seed{a.seed}"
    g.frame.to_csv(f"{stem}.csv", index=False)
    g.answer_key.to_csv(f"{stem}.answer_key.csv", index=False)
    g.events.to_csv(f"{stem}.events.csv", index=False)
    Path(f"{stem}.info.json").write_text(json.dumps(g.info, indent=2))
    print(f"wrote {stem}.csv ({length} rows, {len(g.frame.columns) - 1} features, no label column)")
    print(f"hidden answer key: {stem}.answer_key.csv  ({g.info['events']} events, "
          f"{g.info['anomalous_fraction']:.1%} of steps)")
    print(g.events.to_string(index=False))


if __name__ == "__main__":
    main()
