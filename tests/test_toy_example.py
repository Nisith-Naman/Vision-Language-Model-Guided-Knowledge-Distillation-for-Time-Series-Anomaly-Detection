"""WORKED_EXAMPLE.md stays in sync with the code: the toy script must run and
reproduce its numbers with the project's functions."""

import runpy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def test_toy_example_numbers():
    g = runpy.run_path(str(ROOT / "scripts" / "toy_example.py"), run_name="toy")
    assert np.allclose(g["mu"], [20, 100]) and np.allclose(g["sd"], [2, 5])
    assert list(g["starts"]) == [0, 2, 4]
    assert abs(g["p_T"][1] - 0.7311) < 1e-4 and abs(g["p_win"][1] - 0.6900) < 1e-4
    assert abs(g["total"] - 0.8058) < 1e-4
    assert abs(g["tau"] - 0.301) < 1e-3
    assert np.allclose(g["step_tl"], [0.1, 0.6, 0.235, 0.235, 0.85, 0.815, 0.1, 0.1])
