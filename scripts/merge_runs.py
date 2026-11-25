"""Merge selected models from a partial benchmark run into the main results.

Used when only some variants were re-run (e.g. after fixing the probe's
cross-fitting). Rows, training histories and seed-0 test predictions of the
named models replace those in the main directory; everything else is kept.

    .venv\\Scripts\\python.exe scripts\\merge_runs.py --source experiments/benchmark_v2_probefix ^
        --datasets creditcard,metropt,oil3w --models gru_kd_probe,gru_kd_full,gru_teacher_only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--target", default="experiments/benchmark_v2")
    p.add_argument("--datasets", required=True)
    p.add_argument("--models", required=True)
    a = p.parse_args()
    models = set(a.models.split(","))
    for ds in a.datasets.split(","):
        src, dst = ROOT / a.source / ds, ROOT / a.target / ds
        new = pd.read_csv(src / "metrics.csv", dtype={"seed": str})
        old = pd.read_csv(dst / "metrics.csv", dtype={"seed": str})
        new = new[new.model.isin(models)]
        if new.empty:
            raise SystemExit(f"{ds}: no rows for {models} in {src}")
        pd.concat([old[~old.model.isin(models)], new], ignore_index=True).to_csv(dst / "metrics.csv", index=False)
        h_old = json.loads((dst / "training_histories.json").read_text())
        h_new = json.loads((src / "training_histories.json").read_text())
        h_old = {k: v for k, v in h_old.items() if k.rsplit("_seed", 1)[0] not in models}
        h_old.update({k: v for k, v in h_new.items() if k.rsplit("_seed", 1)[0] in models})
        (dst / "training_histories.json").write_text(json.dumps(h_old, indent=1))
        z_old = dict(np.load(dst / "test_predictions_seed0.npz"))
        z_new = dict(np.load(src / "test_predictions_seed0.npz"))
        assert np.array_equal(z_old["labels"], z_new["labels"]), f"{ds}: test labels differ between runs"
        z_old.update({k: v for k, v in z_new.items() if k.split("__")[0] in models})
        np.savez_compressed(dst / "test_predictions_seed0.npz", **z_old)
        (dst / f"merged_from_{Path(a.source).name}.json").write_text(json.dumps(
            {"source": a.source, "models": sorted(models), "rows": int(len(new))}, indent=2))
        print(f"{ds}: merged {len(new)} rows for {sorted(models)}")


if __name__ == "__main__":
    main()
