"""Controlled v2 benchmark: frozen-CLIP teachers vs. GRU students vs. baselines.

For every dataset the script builds ONE fixed split and evaluates, on the same
validation/test windows:

  Teachers (CLIP ViT-B/32 weights frozen)
    clip_zeroshot      prompt-ensemble zero-shot scores on rendered windows
    clip_probe         logistic probe on frozen CLIP image embeddings
  Established baselines
    isolation_forest   unsupervised, window statistics
    gbdt               supervised gradient-boosted trees, window statistics
  Students (same architecture, same initial weights per seed)
    gru_gt             ground-truth labels only (no teacher)
    gru_kd_zeroshot    + KD from zero-shot CLIP soft labels
    gru_kd_probe       + KD from probe soft labels (cross-fitted)
    gru_kd_full        + KD from probe soft labels + CLIP feature distillation
    gru_teacher_only   window head learns from the teacher only (beta = 0)

Run from the project root:
    .venv\\Scripts\\python.exe scripts\\run_benchmark.py
    .venv\\Scripts\\python.exe scripts\\run_benchmark.py --datasets daphnet --seeds 0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import yaml

from src.evaluation.postprocess import evaluate
from src.experiments.baselines import gbdt_scores, isolation_forest_scores
from src.experiments.data import PreparedData, prepare_dataset
from src.student.model import GRUStudent
from src.teacher.clip_teacher import ANOMALY_PROMPTS, NORMAL_PROMPTS, CLIPEmbedder
from src.teacher.probe import LinearProbeTeacher
from src.teacher.render import render_batch, resolve_style
from src.training.checkpoint import CheckpointMetadata, save_checkpoint
from src.training.trainer import Trainer

# Air Quality (synthetic labels) is excluded from the v2 benchmark for now;
# its loader and results remain in experiments/excluded_airquality.
ALL_DATASETS = ["creditcard", "daphnet", "oil3w", "metropt"]
STUDENTS = ["gru_gt", "gru_kd_zeroshot", "gru_kd_probe", "gru_kd_full", "gru_teacher_only",
            "gru_kd_coop_plain", "gru_kd_coop"]
TEACHERS = ["clip_zeroshot", "clip_probe", "clip_probe_mv", "clip_coop", "clip_coop_mv"]
EXPORT = ("gru_gt", "gru_kd_full", "gru_kd_coop")   # checkpoints written for the interface
LABELS = {
    "clip_zeroshot": "CLIP zero-shot (frozen)",
    "clip_probe": "CLIP linear probe (frozen backbone)",
    "clip_probe_mv": "CLIP linear probe, heatmap + spectrogram",
    "clip_coop": "CLIP learned prompts (CoOp)",
    "clip_coop_mv": "CLIP learned prompts, heatmap + spectrogram",
    "isolation_forest": "Isolation Forest",
    "gbdt": "Gradient-boosted trees",
    "gru_gt": "GRU, ground truth only",
    "gru_kd_zeroshot": "GRU + KD (zero-shot CLIP)",
    "gru_kd_probe": "GRU + KD (probe soft labels)",
    "gru_kd_full": "GRU + KD (soft labels + CLIP features)",
    "gru_teacher_only": "GRU, window head from teacher only (beta=0)",
    "gru_kd_coop_plain": "GRU + KD (learned prompts, 2 views)",
    "gru_kd_coop": "GRU + KD (learned prompts, 2 views, confidence-weighted)",
}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Teacher
# ---------------------------------------------------------------------------

TEACHER_VERSION = "v2.1: block-interleaved cross-fitting; CoOp prompts; heatmap+spectrogram views"


def _embed_view(data: PreparedData, X_by_split: dict, view: str, style: str, clip_value: float,
                out_dir: Path, identity: dict, embedder_holder: dict, tcfg: dict) -> dict:
    """CLIP image embeddings of one rendering view for train-subset/val/test (cached)."""
    from src.teacher.render import render_spectrogram

    name = "clip_embeddings" if view == "heatmap" else f"clip_embeddings_{view}"
    cache, meta_path = out_dir / f"{name}.npz", out_dir / f"{name}.json"
    ident = dict(identity) if view == "heatmap" else {**identity, "style": view}
    if cache.exists() and meta_path.exists() and json.loads(meta_path.read_text()) == ident:
        with np.load(cache) as z:
            log(f"  reusing cached CLIP {view} embeddings")
            return {k: z[k] for k in ("train", "val", "test")}
    if "emb" not in embedder_holder:
        embedder_holder["emb"] = CLIPEmbedder(tcfg["model_name"]).load()
        assert embedder_holder["emb"].is_frozen()
    emb, E, t0 = embedder_holder["emb"], {}, time.time()
    for split, X in X_by_split.items():
        chunks = []
        for s in range(0, len(X), 512):
            imgs = (np.stack([render_spectrogram(w, clip_value) for w in X[s:s + 512]])
                    if view == "spectrogram" else render_batch(X[s:s + 512], style, clip_value))
            chunks.append(emb.encode_images(imgs))
        E[split] = np.concatenate(chunks)
        log(f"  CLIP embedded {view} {split}: {len(X)} windows ({time.time() - t0:.0f}s)")
    np.savez_compressed(cache, **E)
    meta_path.write_text(json.dumps(ident, indent=2))
    return E


def teacher_outputs(data: PreparedData, tcfg: dict, out_dir: Path, embedder_holder: dict) -> dict:
    """All teachers on frozen CLIP: zero-shot, probe (1 and 2 views), learned prompts
    (1 and 2 views). Training-window soft labels are cross-fitted (out-of-fold).
    Results are cached so that parallel student runs share one teacher fit."""
    from src.teacher.prompt_learning import PromptLearningTeacher

    tr, va, te = data.splits["train"], data.splits["val"], data.splits["test"]
    cap = int(tcfg.get("max_train_windows", 8000))
    train_idx = np.arange(len(tr.X)) if len(tr.X) <= cap else np.linspace(0, len(tr.X) - 1, cap).astype(int)
    style = resolve_style(tcfg.get("render_style", "auto"), len(data.feature_names))
    clip_value = float(tcfg.get("render_clip", 4.0))
    identity = {"sha": data.manifest["processed_sha256"], "style": style, "clip": clip_value,
                "model": tcfg["model_name"], "n_train": int(len(train_idx)),
                "n_val": int(len(va.X)), "n_test": int(len(te.X))}
    X_by_split = {"train": tr.X[train_idx], "val": va.X, "test": te.X}
    E = _embed_view(data, X_by_split, "heatmap", style, clip_value, out_dir, identity, embedder_holder, tcfg)
    S = _embed_view(data, X_by_split, "spectrogram", style, clip_value, out_dir, identity, embedder_holder, tcfg)
    if "emb" not in embedder_holder:
        embedder_holder["emb"] = CLIPEmbedder(tcfg["model_name"]).load()
    emb = embedder_holder["emb"]
    y_tr = tr.y_window[train_idx]
    folds, block = int(tcfg.get("probe_folds", 5)), int(tcfg.get("fold_block", 32))

    cache = out_dir / "teacher_outputs.npz"
    cache_meta = out_dir / "teacher_outputs.json"
    t_ident = {**identity, "version": TEACHER_VERSION, "folds": folds, "block": block,
               "coop_epochs": int(tcfg.get("coop_epochs", 12))}
    P, info = {}, {}
    if cache.exists() and cache_meta.exists() and json.loads(cache_meta.read_text()).get("identity") == t_ident:
        with np.load(cache) as z:
            P = {k: z[k] for k in z.files}
        info = json.loads(cache_meta.read_text())["info"]
        log("  reusing cached teacher outputs")
    else:
        t0 = time.time()
        for k in ("train", "val", "test"):
            P[f"zeroshot_{k}"] = emb.zero_shot(E[k], NORMAL_PROMPTS, ANOMALY_PROMPTS)
        both = {k: np.concatenate([E[k], S[k]], 1) for k in E}
        for name, feats in (("probe", E), ("probe_mv", both)):
            probe = LinearProbeTeacher(n_folds=folds, block_size=block)
            P[f"{name}_train"] = probe.fit(feats["train"], y_tr)
            P[f"{name}_val"] = probe.predict_proba(feats["val"])
            P[f"{name}_test"] = probe.predict_proba(feats["test"])
            probe.save(str(out_dir / f"clip_{name}.joblib"))
            info[name] = {"C": probe.C_, "cv_auprc": probe.cv_auprc_}
            log(f"  fitted {name} ({time.time() - t0:.0f}s)")
        for name, views in (("coop", [E]), ("coop_mv", [E, S])):
            coop = PromptLearningTeacher(epochs=int(tcfg.get("coop_epochs", 12)), n_folds=folds, block_size=block)
            P[f"{name}_train"] = coop.fit(emb, np.stack([v["train"] for v in views]), y_tr)
            P[f"{name}_val"] = coop.predict_proba(np.stack([v["val"] for v in views]))
            P[f"{name}_test"] = coop.predict_proba(np.stack([v["test"] for v in views]))
            np.savez_compressed(out_dir / f"clip_{name}.npz", ctx=coop.ctx_)
            info[name] = {"final_train_loss": coop.history_[-1],
                          "views": ["heatmap", "spectrogram"][:len(views)],
                          "nearest_words": coop.nearest_words(emb)}
            log(f"  fitted {name} ({time.time() - t0:.0f}s)")
        np.savez_compressed(cache, **P)
        cache_meta.write_text(json.dumps({"identity": t_ident, "info": info}, indent=2))

    def spread(p_sub):
        full = np.tile(np.array([[0.5, 0.5]], np.float32), (len(tr.X), 1))
        full[train_idx] = p_sub
        return full

    def full_emb(arr):
        out = np.zeros((len(tr.X), arr.shape[1]), np.float32)
        out[train_idx] = arr
        return out

    mv = E["train"] + S["train"]
    mv = mv / np.linalg.norm(mv, axis=1, keepdims=True)
    mask = np.zeros(len(tr.X), bool)
    mask[train_idx] = True
    teachers = {name: {"train": spread(P[f"{name}_train"]), "val": P[f"{name}_val"],
                       "test": P[f"{name}_test"], "train_sub": P[f"{name}_train"]}
                for name in ("zeroshot", "probe", "probe_mv", "coop", "coop_mv")}
    return {"mask": mask, "train_idx": train_idx, "style": style, **teachers,
            "embeddings_train": full_emb(E["train"]), "embeddings_train_mv": full_emb(mv),
            "probe_info": info.get("probe", {}), "info": info}


# ---------------------------------------------------------------------------
# Students
# ---------------------------------------------------------------------------

def build_student(data: PreparedData, scfg: dict, downsample: int, seed: int) -> GRUStudent:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return GRUStudent(
        input_size=len(data.feature_names), hidden_size=int(scfg["hidden_size"]),
        num_layers=int(scfg["num_layers"]), dropout=float(scfg["dropout"]),
        seq_len=data.window_length, bidirectional=bool(scfg["bidirectional"]),
        conv_channels=int(scfg["conv_channels"]), conv_kernel=int(scfg["conv_kernel"]),
        downsample=downsample, pooling=str(scfg["pooling"]), embed_dim=512,
    )


def predict(model: GRUStudent, X: np.ndarray, batch: int = 256):
    w, s = [], []
    for i in range(0, len(X), batch):
        pw, ps = model.predict_proba(torch.as_tensor(X[i:i + batch], dtype=torch.float32))
        w.append(pw[:, 1]); s.append(ps)
    return np.concatenate(w), np.concatenate(s)


def train_student(variant: str, data: PreparedData, teacher: dict, bcfg: dict, downsample: int,
                  seed: int, pos_weight: float, overrides: dict | None = None):
    scfg = {**bcfg["student"], **(overrides or {}).get("student", {})}
    model = build_student(data, scfg, downsample, seed)
    dcfg = dict(bcfg["distillation"])
    mode, t_probs, feat = "student_kd", None, 0.0
    if variant == "gru_gt":
        mode = "student_baseline"
    elif variant == "gru_kd_zeroshot":
        t_probs = teacher["zeroshot"]["train"]
    elif variant == "gru_kd_probe":
        t_probs = teacher["probe"]["train"]
    elif variant == "gru_kd_full":
        t_probs, feat = teacher["probe"]["train"], float(dcfg["lambda_feat"])
    elif variant == "gru_teacher_only":
        t_probs, feat = teacher["probe"]["train"], float(dcfg["lambda_feat"])
        dcfg["beta"] = 0.0
    elif variant in ("gru_kd_coop", "gru_kd_coop_plain"):
        t_probs, feat = teacher["coop_mv"]["train"], float(dcfg["lambda_feat"])
        dcfg["kd_confidence"] = variant == "gru_kd_coop"
    emb_target = (teacher["embeddings_train_mv"] if variant.startswith("gru_kd_coop")
                  else teacher["embeddings_train"])
    dcfg["lambda_feat"] = feat
    dcfg["pos_weight"] = pos_weight
    if mode == "student_baseline":
        dcfg["lambda_kd"] = 0.0
    tcfg = dict(bcfg["training"])
    tcfg.update((overrides or {}).get("training", {}))
    tcfg["optimizer"] = "adamw"
    trainer = Trainer(model=model, config={"mode": mode, "device": "cpu", "training": tcfg, "distillation": dcfg})
    tr, va = data.splits["train"], data.splits["val"]
    hist = trainer.fit(
        tr.X, tr.y_step, tr.y_window, t_probs, va.X, va.y_step, va.y_window, None,
        teacher_mask_train=teacher["mask"] if t_probs is not None else None,
        teacher_embeddings_train=emb_target if feat > 0 else None,
    )
    return trainer.model, hist.to_dict(), {"mode": mode, **dcfg, "training": tcfg, "student": scfg}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def metric_rows(dataset: str, model_key: str, seed, res: dict, extra: dict | None = None) -> list[dict]:
    base = {"dataset": dataset, "model": model_key, "label": LABELS[model_key], "seed": seed}
    rows = []
    for level in ("window", "point"):
        rows.append({**base, "level": level, **{k: v for k, v in res[level].items()}, **(extra or {})})
    rows.append({**base, "level": "event", **res["event"], **(extra or {})})
    return rows


def run_dataset(name: str, cfg: dict, args, embedder_holder: dict) -> pd.DataFrame:
    bcfg = cfg["benchmark"]
    pcfg = bcfg["per_dataset"].get(name, {})
    out_dir = Path(args.output_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"=== {name} ===")
    data = prepare_dataset(name, cfg, max_train_windows=int(pcfg.get("max_train_windows", 0)))
    (out_dir / "manifest.json").write_text(json.dumps(data.manifest, indent=2, default=str))
    if data.label_audit:
        pd.DataFrame(data.label_audit).to_csv(out_dir / "label_audit.csv", index=False)
    for k, v in data.manifest["partitions"].items():
        log(f"  {k}: windows={v['windows']} pos_windows={v['anomalous_windows']} "
            f"point_prev={v['point_prevalence']:.4f}")

    teacher = teacher_outputs(data, bcfg["teacher"], out_dir, embedder_holder)
    va, te, tr = data.splits["val"], data.splits["test"], data.splits["train"]
    L = data.window_length
    rows: list[dict] = []
    preds_to_save: dict = {"labels": np.concatenate([s["labels"] for s in te.sequences]),
                           "seq_lengths": np.array([len(s["labels"]) for s in te.sequences])}

    def add(model_key, seed, vw, tw, vs=None, ts=None, extra=None):
        res = evaluate(va, te, L, vw, tw, vs, ts)
        rows.extend(metric_rows(name, model_key, seed, res, extra))
        if seed in ("-", 0):
            preds_to_save[f"{model_key}__score"] = np.concatenate(res["test_scores"])
            preds_to_save[f"{model_key}__pred"] = np.concatenate(res["test_preds"])
        return res

    # Results are merged into any existing files in out_dir: rows/histories of the
    # models evaluated in this invocation replace older ones; other models are kept.
    prior_rows = pd.read_csv(out_dir / "metrics.csv", dtype={"seed": str}) if (out_dir / "metrics.csv").exists() else None
    hist_path, npz_path = out_dir / "training_histories.json", out_dir / "test_predictions_seed0.npz"
    prior_hist = json.loads(hist_path.read_text()) if hist_path.exists() else {}
    prior_npz = dict(np.load(npz_path)) if npz_path.exists() else {}

    def write_rows():
        new = pd.DataFrame(rows)
        new["seed"] = new["seed"].astype(str)
        if prior_rows is not None:
            new = pd.concat([prior_rows[~prior_rows.model.isin(set(new.model))], new], ignore_index=True)
        new.to_csv(out_dir / "metrics.csv", index=False)

    log("  evaluating teachers and baselines")
    from sklearn.metrics import average_precision_score
    m = teacher["mask"]
    y_sub = tr.y_window[teacher["train_idx"]]
    oof_ap = {}
    for tk in ("zeroshot", "probe", "probe_mv", "coop", "coop_mv"):
        add(f"clip_{tk}", "-", teacher[tk]["val"][:, 1], teacher[tk]["test"][:, 1])
        oof_ap[tk] = float(average_precision_score(y_sub, teacher[tk]["train_sub"][:, 1]))
    if args.with_baselines:
        if_v, if_t = isolation_forest_scores(tr.X, [va.X, te.X])
        add("isolation_forest", "-", if_v, if_t)
        gb_v, gb_t = gbdt_scores(tr.X, tr.y_window, [va.X, te.X])
        add("gbdt", "-", gb_v, gb_t)
    write_rows()

    pos = float(tr.y_step.mean())
    pos_weight = float(np.clip(np.sqrt((1 - pos) / max(pos, 1e-6)), 1.0, 25.0))
    downsample = int(pcfg.get("downsample", 1))
    histories, best = {}, {}
    variants = [v for v in STUDENTS if v in args.variants]
    for seed in args.seeds:
        for variant in variants:
            t0 = time.time()
            model, hist, used_cfg = train_student(variant, data, teacher, bcfg, downsample, seed, pos_weight, pcfg)
            vw, vs = predict(model, va.X)
            tw, ts = predict(model, te.X)
            res = add(variant, seed, vw, tw, vs, ts, extra={"best_epoch": hist["best_epoch"],
                                                          "val_auprc_best": float(np.nanmax(hist["val_auprc"])),
                                                          "train_seconds": round(time.time() - t0, 1)})
            histories[f"{variant}_seed{seed}"] = {**hist, "config": used_cfg}
            val_score = float(np.nanmax(hist["val_auprc"])) if hist["val_auprc"] else 0.0
            if variant in EXPORT and val_score > best.get(variant, (-1,))[0]:
                best[variant] = (val_score, seed, model, res["params"], hist, used_cfg)
            log(f"  {variant:17s} seed={seed} test window AUPRC={res['window']['auprc']:.3f} "
                f"point AUPRC={res['point']['auprc']:.3f} event F1={res['event']['event_f1']:.3f} "
                f"({time.time() - t0:.0f}s, best epoch {hist['best_epoch']})")
            write_rows()

    hist_path.write_text(json.dumps({**prior_hist, **histories}, indent=1))
    np.savez_compressed(npz_path, **{**prior_npz, **preds_to_save})
    with open(out_dir / "teacher_summary.json", "w") as f:
        json.dump({"render_style": teacher["style"], "views": ["heatmap", "spectrogram"],
                   "train_windows_scored": int(m.sum()), "train_windows_total": int(len(m)),
                   "teacher_version": TEACHER_VERSION, "teachers": teacher["info"],
                   "train_oof_window_auprc": oof_ap,
                   "train_oof_window_auprc_probe": oof_ap["probe"],
                   "train_window_auprc_zeroshot": oof_ap["zeroshot"],
                   "train_window_prevalence": float(y_sub.mean()),
                   "pos_weight": pos_weight, "downsample": downsample}, f, indent=2)

    # Deployable checkpoints for the Streamlit interface (best validation seed).
    for variant, (val_score, seed, model, params, hist, used_cfg) in best.items():
        meta = CheckpointMetadata(
            model_config=model.get_config().to_dict(), feature_names=data.feature_names,
            scaler_type="standard", window_length=L, stride=data.stride,
            threshold=float(params.threshold), normal_label=0, anomaly_label=1,
            training_mode="student_kd" if variant != "gru_gt" else "student_baseline",
            dataset_name=f"benchmark_{name}",
            training_result={**hist, "localization": params.to_dict(), "benchmark_variant": variant,
                             "seed": seed, "val_auprc": val_score, "distillation": used_cfg},
            created_at=datetime.now().isoformat(timespec="seconds"),
        )
        prefix = getattr(args, "checkpoint_prefix", "v2")
        save_checkpoint(model, data.scaler, meta, str(Path("checkpoints") / f"{prefix}_{name}_{variant}"))
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, out_dir: Path) -> None:
    keys = ["dataset", "model", "label", "level"]
    num = [c for c in ("prevalence", "auprc", "roc_auc", "precision", "recall", "f1",
                       "event_precision", "event_recall", "event_f1") if c in df]
    agg = df.groupby(keys)[num].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(c).rstrip("_") for c in agg.columns]
    agg.to_csv(out_dir / "summary.csv", index=False)

    paired = []
    for (ds, level), g in df[df.seed != "-"].groupby(["dataset", "level"]):
        metric = "event_f1" if level == "event" else "auprc"
        base = g[g.model == "gru_gt"].set_index("seed")[metric]
        for variant in STUDENTS[1:]:
            v = g[g.model == variant].set_index("seed")[metric]
            common = base.index.intersection(v.index)
            if len(common):
                d = (v[common] - base[common]).astype(float)
                paired.append({"dataset": ds, "level": level, "metric": metric, "variant": variant,
                               "mean_delta_vs_gt": d.mean(), "wins": int((d > 0).sum()), "seeds": len(common)})
    pd.DataFrame(paired).to_csv(out_dir / "paired_vs_gt.csv", index=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", default=",".join(ALL_DATASETS))
    p.add_argument("--seeds", default=None, help="comma separated, default from config")
    p.add_argument("--variants", default=",".join(STUDENTS))
    p.add_argument("--output-dir", default=None)
    p.add_argument("--checkpoint-prefix", default="v2", help="checkpoints/<prefix>_<dataset>_<variant>")
    p.add_argument("--no-baselines", dest="with_baselines", action="store_false",
                   help="skip Isolation Forest / gradient-boosted trees (keep earlier rows)")
    p.add_argument("--threads", type=int, default=max(1, min(12, os.cpu_count() or 1)))
    p.add_argument("--overrides", default=None,
                   help='JSON merged into benchmark.per_dataset[<name>] for every dataset run, '
                        'e.g. {"training": {"learning_rate": 0.001, "ema_decay": 0.99}}')
    args = p.parse_args()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    if args.overrides:
        extra = json.loads(args.overrides)
        for name in args.datasets.split(","):
            pd_cfg = cfg["benchmark"]["per_dataset"].setdefault(name.strip(), {})
            for key, val in extra.items():
                pd_cfg[key] = {**pd_cfg.get(key, {}), **val} if isinstance(val, dict) else val
    args.datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    args.seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else list(cfg["benchmark"]["seeds"])
    args.variants = [v.strip() for v in args.variants.split(",")]
    args.output_dir = args.output_dir or cfg["benchmark"]["output_dir"]
    torch.set_num_threads(args.threads)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    holder: dict = {}
    frames = []
    for name in args.datasets:
        frames.append(run_dataset(name, cfg, args, holder))
    # Merge with any datasets finished in earlier invocations.
    all_frames = []
    for name in ALL_DATASETS:
        f = out / name / "metrics.csv"
        if f.exists():
            all_frames.append(pd.read_csv(f, dtype={"seed": str}))
    df = pd.concat(all_frames, ignore_index=True)
    df.to_csv(out / "all_metrics.csv", index=False)
    summarise(df, out)
    (out / "run_info.json").write_text(json.dumps({
        "finished_utc": datetime.now(timezone.utc).isoformat(), "seeds": args.seeds,
        "datasets": args.datasets, "config": cfg["benchmark"]}, indent=2))
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
