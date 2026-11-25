"""Tests for the v2 teacher, student, objective, and localization components."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.postprocess import (
    LocalizationParams,
    aggregate,
    apply_localization,
    build_timelines,
    event_scores,
    remove_short,
    runs,
    smooth,
    tune_localization,
)
from src.experiments.baselines import gbdt_scores, isolation_forest_scores, window_features
from src.student.losses import TotalLoss, soften_teacher
from src.student.model import GRUStudent, ModelConfig
from src.teacher.probe import LinearProbeTeacher
from src.teacher.render import render_batch, render_window, resolve_style
from src.training.trainer import Trainer


def test_student_downsample_restores_step_length_and_embedding():
    m = GRUStudent(input_size=9, hidden_size=16, num_layers=1, seq_len=256, bidirectional=True,
                   conv_channels=8, downsample=4, pooling="meanmax", embed_dim=512)
    x = torch.randn(3, 256, 9)
    w, s, e = m(x, return_embedding=True)
    assert w.shape == (3, 2) and s.shape == (3, 256) and e.shape == (3, 512)
    cfg = ModelConfig.from_dict(m.get_config().to_dict())
    assert cfg.downsample == 4 and cfg.embed_dim == 512
    # The training-only embedding head is excluded from the deployed size.
    full = sum(p.numel() for p in m.parameters())
    assert m.parameter_count() < full


def test_legacy_student_config_still_builds():
    m = GRUStudent(**{"input_size": 3, "hidden_size": 8, "num_layers": 2, "dropout": 0.1,
                      "seq_len": 20, "bidirectional": False})
    w, s = m(torch.randn(2, 20, 3))
    assert w.shape == (2, 2) and s.shape == (2, 20)


def test_soften_teacher_matches_worked_example():
    soft = soften_teacher(torch.tensor([[0.2, 0.8]]), 2.0)
    assert torch.allclose(soft, torch.tensor([[1 / 3, 2 / 3]]), atol=1e-3)


def test_total_loss_terms_and_mask():
    torch.manual_seed(0)
    loss = TotalLoss(beta=1.0, lambda_kd=1.0, pos_weight=2.0, temperature=2.0, lambda_feat=0.5)
    wl, sl = torch.randn(4, 2), torch.randn(4, 6)
    y_step = torch.zeros(4, 6); y_step[0, 2:4] = 1
    y_win = torch.tensor([1, 0, 0, 0])
    tp = torch.tensor([[0.2, 0.8]] * 4)
    se, te = torch.randn(4, 8), torch.randn(4, 8)
    total, parts = loss(sl, wl, y_step, y_win, tp, return_dict=True,
                        teacher_mask=torch.tensor([1, 1, 0, 0]).bool(), student_embedding=se, teacher_embedding=te)
    expected = parts["loc_loss"] + parts["win_loss"] + parts["kd_loss"] + 0.5 * parts["feat_loss"]
    assert parts["kd_loss"] > 0 and parts["feat_loss"] > 0
    assert abs(float(total) - expected) < 1e-5
    # No teacher rows -> no teacher terms.
    _, parts0 = loss(sl, wl, y_step, y_win, tp, return_dict=True, teacher_mask=torch.zeros(4).bool(),
                     student_embedding=se, teacher_embedding=te)
    assert parts0["kd_loss"] == 0 and parts0["feat_loss"] == 0


def test_trainer_fit_with_teacher_embeddings_and_auprc_selection():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 16, 3)).astype(np.float32)
    ys = np.zeros((64, 16), np.int8); ys[::4, 5:9] = 1
    X[::4, 5:9] += 3
    yw = ys.max(1)
    tp = np.stack([1 - yw * 0.8 - 0.1, yw * 0.8 + 0.1], 1).astype(np.float32)
    emb = rng.normal(size=(64, 512)).astype(np.float32)
    model = GRUStudent(3, hidden_size=8, num_layers=1, seq_len=16, embed_dim=512, conv_channels=4)
    tr = Trainer(model, config={"mode": "student_kd", "device": "cpu",
                                "training": {"epochs": 3, "batch_size": 16, "selection_metric": "val_auprc",
                                             "early_stopping_patience": 3},
                                "distillation": {"lambda_kd": 1.0, "lambda_feat": 0.5}})
    hist = tr.fit(X, ys, yw, tp, X, ys, yw, None, teacher_mask_train=np.ones(64, bool),
                  teacher_embeddings_train=emb)
    assert len(hist.val_auprc) == 3 and all(v > 0 for v in hist.feat_losses)
    assert hist.best_epoch >= 1


def test_render_styles_and_shapes():
    w = np.random.default_rng(0).normal(size=(64, 5)).astype(np.float32)
    assert render_window(w, "lines").shape == (224, 224, 3)
    assert render_window(w, "heatmap").shape == (224, 224, 3)
    assert resolve_style("auto", 5) == "lines" and resolve_style("auto", 29) == "heatmap"
    assert render_batch(np.stack([w, w]), "heatmap").dtype == np.uint8
    # Labels never enter the renderer: identical windows give identical images.
    assert np.array_equal(render_window(w, "heatmap"), render_window(w.copy(), "heatmap"))


def test_probe_teacher_cross_fitting():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(150), np.ones(50)].astype(int)
    rng.shuffle(y)
    emb = rng.normal(size=(200, 16)).astype(np.float32)
    emb[:, 0] += 2.5 * y
    probe = LinearProbeTeacher(C_grid=(0.1, 1.0), n_folds=4)
    oof = probe.fit(emb, y)
    assert oof.shape == (200, 2) and np.allclose(oof.sum(1), 1)
    assert probe.C_ in (0.1, 1.0)
    assert probe.predict_proba(emb).shape == (200, 2)


def test_postprocess_helpers():
    assert runs(np.array([0, 1, 1, 0, 1])) == [(1, 3), (4, 5)]
    assert remove_short(np.array([0, 1, 1, 0, 1]), 2).tolist() == [0, 1, 1, 0, 0]
    assert np.allclose(smooth(np.array([0, 0, 3, 0, 0.0]), 3), [0, 1, 1, 1, 0])
    ev = event_scores([np.array([0, 1, 1, 0, 0, 1])], [np.array([0, 0, 1, 0, 1, 0])])
    assert ev["event_recall"] == 0.5 and ev["event_precision"] == 0.5
    agg = aggregate(np.array([[1.0, 1.0], [0.0, 0.0]]), np.array([0, 1]), 2, 3)
    assert np.allclose(agg, [1.0, 0.5, 0.0])


def test_tune_and_apply_localization():
    labels = np.zeros(100, np.int8); labels[40:60] = 1
    step = labels * 0.8 + 0.1
    seqs = [{"labels": labels, "window_starts": np.arange(0, 81, 10), "window_indices": np.arange(9)}]
    win = np.array([labels[s:s + 20].max() for s in range(0, 81, 10)], float)
    steps = np.stack([step[s:s + 20] for s in range(0, 81, 10)])
    tl = build_timelines(seqs, 20, steps, win)
    p = tune_localization(tl, 20)
    assert isinstance(p, LocalizationParams)
    _, preds = apply_localization(tl, p)
    assert event_scores([labels], preds)["event_f1"] == 1.0


def test_baselines_run():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 10, 2)); y = (rng.random(60) < 0.3).astype(int); X[y == 1] += 2
    assert window_features(X).shape == (60, 12)
    v, t = isolation_forest_scores(X, [X[:10], X[10:20]])
    assert v.shape == (10,) and np.all((v > 0) & (v < 1))
    g = gbdt_scores(X, y, [X])[0]
    assert g.shape == (60,)


def test_trainer_ema_and_augmentation_run():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(48, 16, 3)).astype(np.float32)
    ys = np.zeros((48, 16), np.int8); ys[::3, 4:8] = 1; X[::3, 4:8] += 3
    yw = ys.max(1)
    model = GRUStudent(3, hidden_size=8, num_layers=1, seq_len=16)
    tr = Trainer(model, config={"mode": "student_baseline", "device": "cpu",
                                "training": {"epochs": 2, "batch_size": 16, "selection_metric": "val_auprc",
                                             "ema_decay": 0.9, "augment": {"scale": 0.1, "noise": 0.05}}})
    hist = tr.fit(X, ys, yw, None, X, ys, yw, None)
    assert len(hist.val_auprc) == 2 and tr._ema is not None


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "checkpoints" / "v2_creditcard_gru_kd_full").exists(),
                    reason="v2 benchmark checkpoints not generated")
def test_deployed_checkpoint_reproduces_its_benchmark_metrics():
    """Scoring the deployed Credit Card student on the prepared test split with
    postprocess.evaluate (what the Evaluation page does) must reproduce the
    benchmark's metrics.csv row for that checkpoint's seed and configuration."""
    import pandas as pd
    import yaml
    from src.evaluation.postprocess import evaluate
    from src.experiments.data import prepare_dataset
    from src.training.checkpoint import load_checkpoint

    root = Path(__file__).resolve().parents[1]
    ck = root / "checkpoints" / "v2_creditcard_gru_kd_coop"
    if not (ck / "metadata.json").exists():
        pytest.skip("v2 benchmark checkpoints not generated")
    cfg = yaml.safe_load(open(root / "config.yaml", encoding="utf-8"))
    cwd = Path.cwd()
    try:
        import os
        os.chdir(root)
        data = prepare_dataset("creditcard", cfg)
    finally:
        os.chdir(cwd)
    model, _, meta = load_checkpoint(str(ck))
    tr = meta.training_result
    run = "benchmark_v2_regularized" if float(tr["distillation"]["training"].get("ema_decay", 0) or 0) else "benchmark_v2"
    rec = pd.read_csv(root / "experiments" / run / "creditcard" / "metrics.csv", dtype={"seed": str})
    rec = rec[(rec.model == "gru_kd_coop") & (rec.seed == str(tr["seed"]))]

    def pred(X):
        w, s = [], []
        for i in range(0, len(X), 512):
            pw, ps = model.predict_proba(torch.as_tensor(X[i:i + 512]))
            w.append(pw[:, 1]); s.append(ps)
        return np.concatenate(w), np.concatenate(s)

    vw, vs = pred(data.splits["val"].X)
    tw, ts = pred(data.splits["test"].X)
    res = evaluate(data.splits["val"], data.splits["test"], data.window_length, vw, tw, vs, ts)
    assert np.isclose(res["window"]["auprc"], rec[rec.level == "window"]["auprc"].iloc[0], atol=1e-6)
    assert np.isclose(res["point"]["f1"], rec[rec.level == "point"]["f1"].iloc[0], atol=1e-6)
    assert np.isclose(res["event"]["event_f1"], rec[rec.level == "event"]["event_f1"].iloc[0], atol=1e-6)


def test_probe_folds_spread_a_single_event():
    """A single contiguous anomaly must still give informative out-of-fold labels."""
    rng = np.random.default_rng(0)
    y = np.zeros(1000, int); y[400:460] = 1          # one event, like MetroPT's training failure
    emb = rng.normal(size=(1000, 8)).astype(np.float32); emb[:, 0] += 3 * y
    oof = LinearProbeTeacher(C_grid=(1.0,), n_folds=5, block_size=32).fit(emb, y)
    from sklearn.metrics import average_precision_score
    assert average_precision_score(y, oof[:, 1]) > 0.8


def test_confidence_weighted_kd_downweights_wrong_teacher():
    torch.manual_seed(0)
    wl, sl = torch.randn(4, 2), torch.randn(4, 5)
    ys, yw = torch.zeros(4, 5), torch.tensor([1, 1, 0, 0])
    right = torch.tensor([[0.1, 0.9], [0.1, 0.9], [0.9, 0.1], [0.9, 0.1]])
    wrong = right.flip(-1)
    plain = TotalLoss(lambda_kd=1.0)
    conf = TotalLoss(lambda_kd=1.0, kd_confidence=True)
    _, a = conf(sl, wl, ys, yw, wrong, return_dict=True)
    _, b = plain(sl, wl, ys, yw, wrong, return_dict=True)
    assert a["kd_loss"] < 0.2 * b["kd_loss"]          # wrong teacher -> weight 0.1
    _, c = conf(sl, wl, ys, yw, right, return_dict=True)
    _, d = plain(sl, wl, ys, yw, right, return_dict=True)
    assert abs(c["kd_loss"] - 0.9 * d["kd_loss"]) < 1e-5  # correct teacher -> weight 0.9


def test_prompt_encoder_matches_clip_text_features():
    """With context = its initial phrase, the learnable prompt reproduces CLIP exactly."""
    import os
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("HF_HOME", str(root / "cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from src.teacher.clip_teacher import CLIPEmbedder
    from src.teacher.prompt_learning import INIT_PHRASE, PromptEncoder
    try:
        emb = CLIPEmbedder().load()
    except Exception as exc:  # pragma: no cover - CLIP weights not cached
        pytest.skip(f"CLIP not available: {exc}")
    enc = PromptEncoder(emb.model, emb.tokenizer)
    with torch.no_grad():
        mine = enc.text_features().numpy()
    ref = emb.encode_texts([f"{INIT_PHRASE} normal.", f"{INIT_PHRASE} anomalous."])
    assert np.abs(mine - ref).max() < 1e-4
    assert sum(p.numel() for p in enc.parameters() if p.requires_grad) == enc.n_ctx * 512


def test_threshold_falls_back_to_false_alarm_rate_with_few_events():
    from src.evaluation.postprocess import calibrated_threshold
    rng = np.random.default_rng(0)
    y = np.zeros(10000, int); y[5000:5010] = 1
    s = rng.random(10000) * 0.01; s[5000:5010] = 0.5
    thr_few = calibrated_threshold(y, s, n_events=1)
    assert thr_few < 0.0101 and (s[y == 0] >= thr_few).mean() <= 0.0011
    thr_many = calibrated_threshold(y, s, n_events=10)
    assert 0.01 < thr_many <= 0.5
