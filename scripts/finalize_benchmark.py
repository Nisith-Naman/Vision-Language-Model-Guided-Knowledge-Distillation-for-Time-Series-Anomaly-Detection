"""Post-process the v2 benchmark: summaries, teacher export, demo CSVs, figures.

Run after ``scripts/run_benchmark.py``:
    .venv\\Scripts\\python.exe scripts\\finalize_benchmark.py

Writes into experiments/benchmark_v2/:
    all_metrics.csv, summary.csv, paired_vs_gt.csv    merged results
    RESULTS.md, results_table.tex                    tables for the report
    <dataset>/teacher_params.json                    teacher render + validation-tuned localization
    <dataset>/demo_*.csv                             held-out test sequences for the interface
    figures/*.png                                    plots used in the report
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from scripts.run_benchmark import ALL_DATASETS, LABELS, STUDENTS, TEACHERS, summarise
from src.evaluation.postprocess import build_timelines, runs, tune_localization
from src.experiments.data import prepare_dataset
from src.teacher.probe import LinearProbeTeacher

PRETTY = {"creditcard": "Credit Card", "daphnet": "Daphnet FoG", "airquality": "Air Quality*",
          "oil3w": "3W Oil", "metropt": "MetroPT-3"}
ORDER = [*TEACHERS, "isolation_forest", "gbdt", *STUDENTS]
FINAL_TEACHER, FINAL_STUDENT = "clip_coop_mv", "gru_kd_coop"
COLORS = {"clip_zeroshot": "#FCD34D", "clip_probe": "#F59E0B", "isolation_forest": "#CBD5E1",
          "gbdt": "#64748B", "gru_gt": "#6366F1", "gru_kd_zeroshot": "#FCA5A5",
          "gru_kd_probe": "#F87171", "gru_kd_full": "#DC2626", "gru_teacher_only": "#10B981",
          "clip_probe_mv": "#D97706", "clip_coop": "#FB923C", "clip_coop_mv": "#C2410C",
          "gru_kd_coop_plain": "#BE185D", "gru_kd_coop": "#7F1D1D"}


def merged(bench: Path) -> pd.DataFrame:
    frames = [pd.read_csv(bench / d / "metrics.csv", dtype={"seed": str})
              for d in ALL_DATASETS if (bench / d / "metrics.csv").exists()]
    df = pd.concat(frames, ignore_index=True)
    df["label"] = df["model"].map(LABELS)  # keep display names current
    df.to_csv(bench / "all_metrics.csv", index=False)
    summarise(df, bench)
    return df


def export_teacher(name: str, cfg: dict, bench: Path, data) -> None:
    """Teacher used by the interface: learned prompts on heatmap + spectrogram views
    (falls back to the heatmap probe). Localization is tuned on validation only."""
    out = bench / name
    va = data.splits["val"]
    tz = out / "teacher_outputs.npz"
    if tz.exists() and (out / "clip_coop_mv.npz").exists():
        with np.load(tz) as z:
            val_p = z["coop_mv_val"][:, 1]
        teacher, extra = "coop_mv", {"views": ["heatmap", "spectrogram"], "context_file": "clip_coop_mv.npz"}
    else:
        with np.load(out / "clip_embeddings.npz") as z:
            e_val = z["val"]
        probe = LinearProbeTeacher.load(str(out / "clip_probe.joblib"))
        val_p = probe.predict_proba(e_val)[:, 1]
        teacher, extra = "probe", {"probe_C": probe.C_}
    params = tune_localization(build_timelines(va.sequences, data.window_length, None, val_p),
                               data.window_length, has_step_head=False)
    meta = json.loads((out / "clip_embeddings.json").read_text())
    (out / "teacher_params.json").write_text(json.dumps({
        "teacher": teacher, "model_name": cfg["benchmark"]["teacher"]["model_name"],
        "render_style": meta["style"], "render_clip": meta["clip"], "localization": params.to_dict(), **extra,
        "note": "Threshold, smoothing and minimum event length were tuned on validation windows only.",
    }, indent=2))


def export_demos(name: str, data, bench: Path, max_rows: int = 4000, n: int = 2) -> list[Path]:
    te = data.splits["test"]
    written = []
    seqs = sorted([s for s in te.sequences if s["labels"].sum() > 0], key=lambda s: -s["labels"].sum())
    for i, seq in enumerate(seqs[:n]):
        labels = seq["labels"]
        total = len(labels)
        if total > max_rows:
            first = int(np.flatnonzero(labels)[0])
            start = int(np.clip(first - max_rows // 3, 0, total - max_rows))
            sl = slice(start, start + max_rows)
        else:
            sl = slice(0, total)
        raw = data.scaler.inverse_transform(seq["values"][sl].astype(np.float64))
        df = pd.DataFrame(raw, columns=data.feature_names)
        ts = seq.get("timestamps")
        if ts is not None and len(ts) == total:
            df.insert(0, "timestamp", pd.to_datetime(np.asarray(ts)[sl], errors="coerce"))
        df["label"] = labels[sl]
        path = bench / name / f"demo_{i + 1}_{str(seq['group_id']).replace('/', '_').replace('.csv', '')[:40]}.csv"
        df.to_csv(path, index=False)
        written.append(path)
    return written


def fig_auprc(summary: pd.DataFrame, figs: Path, level: str) -> None:
    datasets = [d for d in ALL_DATASETS if d in set(summary.dataset)]
    models = [m for m in ORDER if m in set(summary.model)]
    fig, ax = plt.subplots(figsize=(12, 4.2))
    width = 0.8 / len(models)
    for j, m in enumerate(models):
        vals, errs = [], []
        for d in datasets:
            r = summary[(summary.dataset == d) & (summary.model == m) & (summary.level == level)]
            vals.append(float(r["auprc_mean"].iloc[0]) if len(r) else np.nan)
            errs.append(float(r["auprc_std"].iloc[0]) if len(r) and pd.notna(r["auprc_std"].iloc[0]) else 0)
        ax.bar(np.arange(len(datasets)) + (j - len(models) / 2 + 0.5) * width, vals, width, yerr=errs,
               color=COLORS[m], label=LABELS[m], capsize=2, edgecolor="white", linewidth=0.4)
    for i, d in enumerate(datasets):
        r = summary[(summary.dataset == d) & (summary.level == level)]
        prev = float(r["prevalence_mean"].dropna().iloc[0])
        ax.hlines(prev, i - 0.42, i + 0.42, colors="black", linestyles="dashed", linewidth=1)
    ax.set_xticks(np.arange(len(datasets)), [PRETTY[d] for d in datasets])
    ax.set_ylabel(f"Test {level}-level AUPRC")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{level.capitalize()}-level AUPRC on held-out test data (dashed line = anomaly prevalence = random detector)")
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    fig.tight_layout()
    fig.savefig(figs / f"auprc_{level}.png", dpi=160)
    plt.close(fig)


def fig_training(name: str, bench: Path, figs: Path) -> None:
    hist = json.loads((bench / name / "training_histories.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    student = f"{FINAL_STUDENT}_seed0" if f"{FINAL_STUDENT}_seed0" in hist else "gru_kd_full_seed0"
    for key, color, lab in (("gru_gt_seed0", COLORS["gru_gt"], "GRU, labels only"),
                            (student, COLORS[FINAL_STUDENT], "GRU + CLIP KD")):
        if key not in hist:
            continue
        h = hist[key]
        ep = np.arange(1, len(h["train_losses"]) + 1)
        axes[0].plot(ep, h["train_losses"], color=color, label=f"{lab} (train)")
        axes[0].plot(ep, h["val_losses"], color=color, ls="--", label=f"{lab} (val, supervised terms)")
        axes[1].plot(ep, h["val_auprc"], color=color, label=lab)
        axes[1].axvline(h["best_epoch"], color=color, ls=":", lw=1)
        if key.startswith("gru_kd"):
            axes[2].plot(ep, h["loc_losses"], label="step BCE")
            axes[2].plot(ep, h["window_losses"], label="window CE")
            axes[2].plot(ep, h["kd_losses"], label="KD (T²·KL)")
            axes[2].plot(ep, h["feat_losses"], label="feature (1-cos)")
    axes[0].set_title("Total loss"); axes[1].set_title("Validation AUPRC (model selection)")
    axes[2].set_title("KD student loss components")
    for a in axes:
        a.set_xlabel("epoch"); a.legend(fontsize=7)
    fig.suptitle(f"{PRETTY[name]}: training curves (seed 0)")
    fig.tight_layout()
    fig.savefig(figs / f"training_{name}.png", dpi=150)
    plt.close(fig)


def fig_localization(name: str, data, bench: Path, figs: Path, max_len: int = 3000) -> None:
    z = np.load(bench / name / "test_predictions_seed0.npz")
    lens = z["seq_lengths"]
    offsets = np.r_[0, np.cumsum(lens)]
    te = data.splits["test"]
    # Prefer a sequence with both normal and anomalous steps (context for the plot).
    mixed = [min(int(s["labels"].sum()), int((1 - s["labels"]).sum())) for s in te.sequences]
    i = int(np.argmax(mixed)) if max(mixed) > 0 else int(np.argmax([s["labels"].sum() for s in te.sequences]))
    seq = te.sequences[i]
    labels = seq["labels"]
    total = len(labels)
    first = int(np.flatnonzero(labels)[0]) if labels.sum() else 0
    start = int(np.clip(first - max_len // 3, 0, max(0, total - max_len)))
    sl = slice(start, min(total, start + max_len))
    t = np.arange(sl.start, sl.stop)
    teacher_key = FINAL_TEACHER if f"{FINAL_TEACHER}__score" in z else "clip_probe"
    student_key = FINAL_STUDENT if f"{FINAL_STUDENT}__score" in z else "gru_kd_full"
    models = [(teacher_key, "Teacher (frozen CLIP)", COLORS[teacher_key]),
              ("gru_gt", "GRU labels only", COLORS["gru_gt"]),
              (student_key, "GRU + CLIP KD (student)", COLORS[student_key])]
    fig, axes = plt.subplots(3, 1, figsize=(12, 6.5), sharex=True, gridspec_kw={"height_ratios": [3, 2, 1.2]})
    vals = seq["values"][sl]
    show = min(5, vals.shape[1])
    for f in range(show):
        axes[0].plot(t, np.clip(vals[:, f], -6, 6) + 0 * f, lw=0.7, label=data.feature_names[f])
    for s, e in runs(labels[sl]):
        for a in axes[:2]:
            a.axvspan(t[s], t[min(e, len(t) - 1)], color="#94A3B8", alpha=0.3, lw=0)
    axes[0].set_ylabel("standardised")
    axes[0].legend(fontsize=7, ncol=show, loc="upper right")
    for row, (key, lab, color) in enumerate(models):
        if f"{key}__score" not in z:
            continue
        sc = z[f"{key}__score"][offsets[i]:offsets[i + 1]][sl]
        pr = z[f"{key}__pred"][offsets[i]:offsets[i + 1]][sl]
        axes[1].plot(t, sc, color=color, lw=1, label=lab)
        for s, e in runs(pr):
            axes[2].axvspan(t[s], t[min(e, len(t) - 1)], ymin=(row + 1) / 4 + 0.02, ymax=(row + 2) / 4 - 0.02,
                            color=color, lw=0)
    for s, e in runs(labels[sl]):
        axes[2].axvspan(t[s], t[min(e, len(t) - 1)], ymin=0.02, ymax=0.23, color="#94A3B8", lw=0)
    axes[1].set_ylabel("anomaly score"); axes[1].legend(fontsize=7, loc="upper right")
    axes[2].set_yticks([0.125, 0.375, 0.625, 0.875], ["truth", "teacher", "GRU labels", "GRU+KD"])
    axes[2].set_xlabel("timestep in test sequence")
    fig.suptitle(f"{PRETTY[name]}: held-out localization (grey = ground truth) — sequence {seq['group_id']}")
    fig.tight_layout()
    fig.savefig(figs / f"localization_{name}.png", dpi=150)
    plt.close(fig)


def fig_daphnet_before_after(summary: pd.DataFrame, figs: Path) -> dict:
    v1 = pd.read_csv(ROOT / "experiments" / "daphnet_controlled_v1" / "comparison_metrics.csv")
    v1t = v1.iloc[[0] + list(range(2, len(v1)))]  # drop the validation row of the teacher
    get1 = lambda m, lvl, col="auprc": float(v1t[(v1t.model == m) & (v1t.level == lvl)][col].iloc[0])
    d = summary[summary.dataset == "daphnet"]
    get2 = lambda m, lvl, col="auprc": float(d[(d.model == m) & (d.level == lvl)][f"{col}_mean"].iloc[0])
    rows = {
        "Teacher window AUPRC": (get1("Frozen CLIP", "window"), get2(FINAL_TEACHER, "window")),
        "GRU (labels) window AUPRC": (get1("GRU supervised", "window"), get2("gru_gt", "window")),
        "GRU+KD window AUPRC": (get1("GRU + CLIP KD", "window"), get2(FINAL_STUDENT, "window")),
        "GRU+KD point AUPRC": (get1("GRU + CLIP KD", "point"), get2(FINAL_STUDENT, "point")),
        "GRU+KD event F1": (get1("GRU + CLIP KD", "event", "event_f1"), get2(FINAL_STUDENT, "event", "event_f1")),
    }
    fig, ax = plt.subplots(figsize=(8, 3.4))
    y = np.arange(len(rows))
    ax.barh(y + 0.2, [r[0] for r in rows.values()], 0.4, color="#CBD5E1", label="v1 (primary assessment)")
    ax.barh(y - 0.2, [r[1] for r in rows.values()], 0.4, color="#DC2626", label="v2 (this revision)")
    for yi, (a, b) in zip(y, rows.values()):
        ax.text(b + 0.01, yi - 0.2, f"{b:.3f}", va="center", fontsize=8)
        ax.text(a + 0.01, yi + 0.2, f"{a:.3f}", va="center", fontsize=8, color="#475569")
    ax.set_yticks(y, list(rows)); ax.invert_yaxis(); ax.set_xlim(0, 1.05)
    ax.set_title("Daphnet FoG, identical subject-held-out test split (prevalence 0.048 windows / 0.022 steps)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(figs / "daphnet_before_after.png", dpi=160)
    plt.close(fig)
    return {k: {"v1": a, "v2": b} for k, (a, b) in rows.items()}


def fmt(r, col):
    if r is None or len(r) == 0 or pd.isna(r[f"{col}_mean"].iloc[0]):
        return "–"
    m, s = r[f"{col}_mean"].iloc[0], r[f"{col}_std"].iloc[0]
    return f"{m:.3f}" + (f" ± {s:.3f}" if pd.notna(s) and s > 0 else "")


def write_tables(summary: pd.DataFrame, paired: pd.DataFrame, bench: Path, manifests: dict, before_after: dict | None,
                 teacher_info: dict) -> None:
    lines = ["# v2 benchmark results", "",
             "All numbers are on the held-out **test** split. Thresholds, smoothing, gating and minimum event length "
             "were chosen on validation data only. Students: mean ± std over seeds 0, 1, 2 with identical splits and "
             "identical initial weights per seed. AUPRC of a random detector equals the prevalence.", ""]
    for d in [d for d in ALL_DATASETS if d in set(summary.dataset)]:
        s = summary[summary.dataset == d]
        part = manifests[d]["partitions"]["test"]
        lines += [f"## {PRETTY[d]}", "",
                  f"Split: {manifests[d]['split_method']}. Test windows: {part['windows']} "
                  f"({part['window_prevalence']:.3f} anomalous); test timesteps: {part['timesteps']} "
                  f"({part['point_prevalence']:.4f} anomalous).", ""]
        ti = teacher_info.get(d, {})
        if ti:
            oof = ti.get("train_oof_window_auprc", {"probe": ti.get("train_oof_window_auprc_probe", float("nan")),
                                                    "zeroshot": ti.get("train_window_auprc_zeroshot", float("nan"))})
            lines += ["Teacher soft-label quality on the training windows it labels (out-of-fold window AUPRC; "
                      f"prevalence {ti['train_window_prevalence']:.3f}): "
                      + ", ".join(f"{k} {v:.3f}" for k, v in oof.items()) + ".", ""]
        lines += ["| Model | Window AUPRC | Window F1 | Point AUPRC | Point F1 | Event F1 |",
                  "|---|---|---|---|---|---|"]
        for m in [m for m in ORDER if m in set(s.model)]:
            g = lambda lvl: s[(s.model == m) & (s.level == lvl)]
            lines.append(f"| {LABELS[m]} | {fmt(g('window'), 'auprc')} | {fmt(g('window'), 'f1')} | "
                         f"{fmt(g('point'), 'auprc')} | {fmt(g('point'), 'f1')} | {fmt(g('event'), 'event_f1')} |")
        lines.append("")
    if len(paired):
        lines += ["## Paired comparison with the ground-truth-only GRU", "",
                  "Mean change over seeds (same split, same initial weights) and the number of seeds where the "
                  "variant beats the ground-truth-only GRU.", "",
                  "| Dataset | Level | Variant | Mean Δ | Wins |", "|---|---|---|---|---|"]
        for _, r in paired.sort_values(["dataset", "level", "variant"]).iterrows():
            lines.append(f"| {PRETTY[r.dataset]} | {r.level} ({r.metric}) | {LABELS[r.variant]} | "
                         f"{r.mean_delta_vs_gt:+.3f} | {r.wins}/{r.seeds} |")
        lines.append("")
    if before_after:
        lines += ["## Daphnet: primary assessment (v1) vs this revision (v2)", "", "| Metric | v1 | v2 |", "|---|---|---|"]
        for k, v in before_after.items():
            lines.append(f"| {k} | {v['v1']:.3f} | {v['v2']:.3f} |")
        lines.append("")
    if "metropt" in set(summary.dataset):
        lines += ["## Caveat on MetroPT-3 (see `metropt/AUDIT.md`, `scripts/audit_metropt.py`)", "",
                  "No leakage was found (disjoint sequences and events; no near-duplicate windows). The test set has "
                  "only two failures and they are visually obvious: the window mean of one sensor (DV_pressure, chosen "
                  "on training data) already reaches test window AUPRC 0.917. Per event, the single-view learned-prompt "
                  "teacher scores 1.000 on failure 3 and 0.498 on failure 4. Block-bootstrap 95% intervals: learned "
                  "prompts 0.50-1.00, probe 0.09-1.00. Read MetroPT numbers as evidence of generalisation from one "
                  "subtle training failure to two obvious ones, not as near-perfect detection.", ""]
    lines += ["Air Quality (synthetic injected labels) is excluded from this benchmark for now; "
              "its runs are kept in `experiments/excluded_airquality/`."]
    (bench / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")

    # LaTeX: compact table of window/point AUPRC and event F1 for key models.
    key_models = ["clip_zeroshot", "clip_probe", "clip_coop_mv", "gbdt", "gru_gt", "gru_kd_full", "gru_kd_coop"]
    tex = ["\\begin{tabular}{llccc}", "\\toprule",
           "Dataset (test prev.) & Model & Window AUPRC & Point AUPRC & Event F1 \\\\", "\\midrule"]
    for d in [d for d in ALL_DATASETS if d in set(summary.dataset)]:
        s = summary[summary.dataset == d]
        prev = manifests[d]["partitions"]["test"]
        first = True
        for m in [m for m in key_models if m in set(s.model)]:
            g = lambda lvl: s[(s.model == m) & (s.level == lvl)]
            name = f"{PRETTY[d].replace('*', '$^*$')} ({prev['window_prevalence']:.3f}/{prev['point_prevalence']:.3f})" if first else ""
            first = False
            cells = [fmt(g("window"), "auprc"), fmt(g("point"), "auprc"), fmt(g("event"), "event_f1")]
            cells = [c.replace("±", "$\\pm$") for c in cells]
            tex.append(f"{name} & {LABELS[m]} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
        tex.append("\\midrule")
    tex[-1] = "\\bottomrule"
    tex.append("\\end{tabular}")
    (bench / "results_table.tex").write_text("\n".join(tex), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bench-dir", default=None)
    args = p.parse_args()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    bench = Path(args.bench_dir or cfg["benchmark"]["output_dir"])
    figs = bench / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    merged(bench)
    summary = pd.read_csv(bench / "summary.csv")
    paired = pd.read_csv(bench / "paired_vs_gt.csv") if (bench / "paired_vs_gt.csv").stat().st_size > 2 else pd.DataFrame()
    manifests, teacher_info = {}, {}
    finished = [d for d in ALL_DATASETS if (bench / d / "teacher_summary.json").exists()]
    summary = summary[summary.dataset.isin(finished)]
    if len(paired):
        paired = paired[paired.dataset.isin(finished)]
    for name in finished:
        pcfg = cfg["benchmark"]["per_dataset"].get(name, {})
        data = prepare_dataset(name, cfg, max_train_windows=int(pcfg.get("max_train_windows", 0)))
        manifests[name] = data.manifest
        teacher_info[name] = json.loads((bench / name / "teacher_summary.json").read_text())
        export_teacher(name, cfg, bench, data)
        demos = export_demos(name, data, bench)
        fig_training(name, bench, figs)
        fig_localization(name, data, bench, figs)
        print(f"{name}: teacher params, {len(demos)} demo CSVs, figures", flush=True)
    for level in ("window", "point"):
        fig_auprc(summary, figs, level)
    before_after = fig_daphnet_before_after(summary, figs) if "daphnet" in manifests else None
    write_tables(summary, paired, bench, manifests, before_after, teacher_info)
    print(f"wrote {bench / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
