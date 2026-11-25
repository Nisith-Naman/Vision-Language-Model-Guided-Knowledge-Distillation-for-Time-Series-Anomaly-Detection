"""Shared, cached access to the final pipeline for the Streamlit pages.

Every page reads the same objects, so the numbers on one page are the numbers
used on the next:

* ``prepared(name)``  the fixed train/val/test windows of a dataset
  (``src.experiments.data.prepare_dataset``, the exact function the benchmark used);
* ``raw_series(name)`` the dataset as loaded from its CSV/raw files, before
  imputation and scaling (for the data-pipeline walkthrough);
* ``teacher(name)``   the frozen-CLIP teachers' cached outputs written by
  ``scripts/run_benchmark.py`` (no CLIP forward pass needed), checked against
  the prepared data's hash;
* ``checkpoint(name, key)`` a deployed student;
* ``embedder()``      frozen CLIP ViT-B/32, loaded once, for the live
  per-window teacher walkthrough.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import streamlit as st
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

BENCH = ROOT / "experiments" / "benchmark_v2"
REG = ROOT / "experiments" / "benchmark_v2_regularized"
CKPT = ROOT / "checkpoints"
FIGS = ROOT / "final_ie643_report" / "figures"

DATASETS = ["creditcard", "daphnet", "oil3w", "metropt"]
PRETTY = {"creditcard": "Credit Card Fraud", "daphnet": "Daphnet FoG", "oil3w": "Petrobras 3W",
          "metropt": "MetroPT-3"}

INFO: Dict[str, Dict[str, str]] = {
    "creditcard": {
        "domain": "card payments (fraud)",
        "raw": "one CSV, `data/creditcard.csv`: 284,807 transactions × (Time, V1–V28, Amount, Class)",
        "sequence": "the whole file is one sequence ordered by `Time`; each row is one transaction",
        "label": "`Class` = 1 (fraud) → timestep label 1",
        "split": "chronological 60 / 20 / 20 % of the rows",
    },
    "daphnet": {
        "domain": "wearable accelerometers, Parkinson's freezing of gait",
        "raw": "one text file per subject and run (`S01R01.txt` …), 64 Hz: time, 9 acceleration axes, annotation",
        "sequence": "rows with annotation 0 (not part of the experiment) are dropped; each uninterrupted stretch of a subject-run file is one sequence",
        "label": "annotation 2 (freeze) → 1, annotation 1 (no freeze) → 0",
        "split": "by subject: 7 subjects train, S02 validation, S04 + S07 test",
    },
    "oil3w": {
        "domain": "offshore oil wells (Petrobras 3W)",
        "raw": "one CSV per well instance (8 pressures, temperatures and flows) at 1 Hz, averaged over 10 s",
        "sequence": "each instance is one sequence; 364 instances (real wells first, no hand-drawn ones)",
        "label": "`class` ≠ 0 (fault or transient) → 1",
        "split": "by instance: whole instances go to train, validation or test",
    },
    "metropt": {
        "domain": "train air-compressor (MetroPT-3)",
        "raw": "one CSV, `data/MetroPT3(AirCompressor).csv`: 15 sensors every 10 s, resampled to 1 minute (analog: median, digital: max)",
        "sequence": "the recording is cut at acquisition gaps longer than 5 minutes; each gap-free block is one sequence",
        "label": "timestamps inside a published failure interval → 1",
        "split": "by failure: failure 1 train, 2 validation, 3 + 4 test; normal blocks split chronologically",
    },
}

MODEL_LABELS = {
    "teacher": "Teacher: frozen CLIP + learned prompts (2 views)",
    "gru_gt": "GRU trained on labels only",
    "gru_kd_coop": "GRU + CLIP distillation (proposed)",
}


@st.cache_resource(show_spinner=False)
def config() -> Dict[str, Any]:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dataset_settings(name: str) -> Dict[str, Any]:
    cfg = config()
    return {**cfg["datasets"][name], **cfg["benchmark"]["per_dataset"].get(name, {})}


@st.cache_resource(show_spinner="Preparing the dataset (load → split → impute → scale → window) ...")
def prepared(name: str):
    from src.experiments.data import prepare_dataset

    pcfg = config()["benchmark"]["per_dataset"].get(name, {})
    return prepare_dataset(name, config(), max_train_windows=int(pcfg.get("max_train_windows", 0)))


@st.cache_resource(show_spinner="Loading the raw dataset files ...")
def raw_series(name: str):
    from src.data.loader import load_dataset

    return load_dataset(name, dict(config()["datasets"][name]))


def pos_weight(data) -> float:
    """Positive-class weight of the step BCE: clip(sqrt(n_neg / n_pos), 1, 25) on training steps."""
    pos = float(data.splits["train"].y_step.mean())
    return float(np.clip(np.sqrt((1 - pos) / max(pos, 1e-6)), 1.0, 25.0))


@st.cache_resource(show_spinner="Loading the cached teacher outputs ...")
def teacher(name: str) -> Optional[Dict[str, Any]]:
    """Cached outputs of all frozen-CLIP teachers for the prepared windows.

    ``train_sub`` are the out-of-fold (cross-fitted) probabilities of the
    ``train_idx`` windows the teacher scored; ``train`` spreads them over all
    training windows (unscored windows get 0.5/0.5 and ``mask`` False).
    """
    d = BENCH / name
    meta_path, npz_path = d / "teacher_outputs.json", d / "teacher_outputs.npz"
    if not (meta_path.exists() and npz_path.exists()):
        return None
    data = prepared(name)
    meta = json.loads(meta_path.read_text())
    ident = meta["identity"]
    if ident["sha"] != data.manifest["processed_sha256"]:
        st.error(f"{PRETTY[name]}: cached teacher outputs belong to different prepared data; rerun the benchmark.")
        return None
    tr = data.splits["train"]
    n = int(ident["n_train"])
    train_idx = np.arange(len(tr.X)) if n >= len(tr.X) else np.linspace(0, len(tr.X) - 1, n).astype(int)
    mask = np.zeros(len(tr.X), bool)
    mask[train_idx] = True
    with np.load(npz_path) as z:
        P = {k: z[k] for k in z.files}
    out: Dict[str, Any] = {"train_idx": train_idx, "mask": mask, "info": meta.get("info", {}),
                           "identity": ident}
    for key in ("zeroshot", "probe", "probe_mv", "coop", "coop_mv"):
        full = np.tile(np.array([[0.5, 0.5]], np.float32), (len(tr.X), 1))
        full[train_idx] = P[f"{key}_train"]
        out[key] = {"train_sub": P[f"{key}_train"], "train": full, "val": P[f"{key}_val"], "test": P[f"{key}_test"]}
    for view, fname in (("heatmap", "clip_embeddings.npz"), ("spectrogram", "clip_embeddings_spectrogram.npz")):
        if (d / fname).exists():
            with np.load(d / fname) as z:
                out[f"emb_{view}"] = {k: z[k] for k in ("train", "val", "test")}
    if "emb_heatmap" in out and "emb_spectrogram" in out:
        mv = out["emb_heatmap"]["train"] + out["emb_spectrogram"]["train"]
        mv = mv / np.linalg.norm(mv, axis=1, keepdims=True)
        full = np.zeros((len(tr.X), mv.shape[1]), np.float32)
        full[train_idx] = mv
        out["emb_train_mv"] = full
    if (d / "clip_coop_mv.npz").exists():
        out["ctx"] = np.load(d / "clip_coop_mv.npz")["ctx"]
    return out


@st.cache_resource(show_spinner="Loading a trained student ...")
def checkpoint(name: str, key: str):
    from src.training.checkpoint import load_checkpoint

    path = CKPT / f"v2_{name}_{key}"
    if not (path / "metadata.json").exists():
        return None
    model, scaler, meta = load_checkpoint(str(path))
    return model.eval(), meta


@st.cache_resource(show_spinner="Loading frozen CLIP ViT-B/32 (151M parameters) ...")
def embedder():
    from src.teacher.clip_teacher import CLIPEmbedder

    return CLIPEmbedder(config()["benchmark"]["teacher"]["model_name"]).load()


def predict(model, X: np.ndarray, batch: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """Window anomaly probability (N,) and per-step anomaly scores (N, L)."""
    w, s = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), batch):
            pw, ps = model.predict_proba(torch.as_tensor(X[i:i + batch], dtype=torch.float32))
            w.append(pw[:, 1])
            s.append(ps)
    return np.concatenate(w), np.concatenate(s)


def benchmark_record(name: str, key: str, seed: Any, config_name: str = "base"):
    """Rows of the benchmark's metrics.csv for one model and seed (for consistency checks)."""
    import pandas as pd

    path = (REG if config_name == "regularized" else BENCH) / name / "metrics.csv"
    if not path.exists():
        return None
    m = pd.read_csv(path, dtype={"seed": str})
    rows = m[(m.model == key) & (m.seed == str(seed))]
    return rows if len(rows) else None


def deployed_config(meta) -> str:
    """'regularized' when the checkpoint was trained with weight EMA, else 'base'."""
    tr = ((meta.training_result or {}).get("distillation") or {}).get("training", {})
    return "regularized" if float(tr.get("ema_decay", 0) or 0) > 0 else "base"


REGULARIZED = {"training": {"learning_rate": 1e-3, "early_stopping_patience": 6, "ema_decay": 0.99,
                            "augment": {"scale": 0.15, "noise": 0.05}},
               "student": {"dropout": 0.3}}

VARIANTS = {"gru_gt": "GRU, labels only (no teacher)",
            "gru_kd_coop": "GRU + CLIP distillation (proposed: learned prompts, 2 views, confidence-weighted)"}


def training_setup(name: str, variant: str, config_name: str = "base") -> Dict[str, Any]:
    """The exact model/loss/optimiser settings the benchmark used for this variant
    (scripts/run_benchmark.py → train_student), optionally with the regularized recipe."""
    cfg = config()["benchmark"]
    data = prepared(name)
    over = REGULARIZED if config_name == "regularized" else {}
    student = {**cfg["student"], **over.get("student", {})}
    training = {**cfg["training"], **over.get("training", {}), "optimizer": "adamw"}
    dist = dict(cfg["distillation"])
    dist["pos_weight"] = pos_weight(data)
    if variant == "gru_gt":
        mode, dist["lambda_kd"], dist["lambda_feat"] = "student_baseline", 0.0, 0.0
    else:
        mode, dist["kd_confidence"] = "student_kd", True
    return {"mode": mode, "student": student, "training": training, "distillation": dist,
            "downsample": int(dataset_settings(name).get("downsample", 1))}


def build_student(name: str, setup: Dict[str, Any], seed: int):
    import random

    from src.student.model import GRUStudent

    data = prepared(name)
    s = setup["student"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return GRUStudent(input_size=len(data.feature_names), hidden_size=int(s["hidden_size"]),
                      num_layers=int(s["num_layers"]), dropout=float(s["dropout"]), seq_len=data.window_length,
                      bidirectional=bool(s["bidirectional"]), conv_channels=int(s["conv_channels"]),
                      conv_kernel=int(s["conv_kernel"]), downsample=setup["downsample"],
                      pooling=str(s["pooling"]), embed_dim=512)


def train_live(name: str, variant: str, config_name: str, seed: int, max_epochs: int, epoch_callback=None):
    """Train one student exactly as the benchmark does, reporting every epoch."""
    from src.training.trainer import Trainer

    data = prepared(name)
    setup = training_setup(name, variant, config_name)
    setup["training"]["epochs"] = int(max_epochs)
    model = build_student(name, setup, seed)
    t = teacher(name)
    kd = variant != "gru_gt"
    trainer = Trainer(model=model, config={"mode": setup["mode"], "device": "cpu", "training": setup["training"],
                                           "distillation": setup["distillation"]})
    tr, va = data.splits["train"], data.splits["val"]
    hist = trainer.fit(tr.X, tr.y_step, tr.y_window, t["coop_mv"]["train"] if kd else None,
                       va.X, va.y_step, va.y_window, None,
                       teacher_mask_train=t["mask"] if kd else None,
                       teacher_embeddings_train=t["emb_train_mv"] if kd else None,
                       epoch_callback=epoch_callback)
    return trainer.model.eval(), hist, setup


def selected_dataset() -> str:
    return st.session_state.get("dataset", DATASETS[0])
