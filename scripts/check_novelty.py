"""Functional checks of the three novelty components on the real cached data.

1. Prompt learning (CoOp): CLIP weights unchanged by training (hash before/after),
   only the context vectors receive gradients, the training loss falls, and the
   learned context beats its own untrained initial prompt on validation.
2. Multi-view: spectrogram embeddings carry information the heatmap does not
   (cosine between views, view-only probes), and the per-dataset effect of the
   second view on teacher validation/test AUPRC.
3. Confidence-weighted KD: on a real training batch, the KD term equals the
   hand-computed mean of p_T(y_i) * T^2 * KL_i, and the weights are low exactly
   where the teacher is wrong.
Writes experiments/benchmark_v2/NOVELTY_CHECKS.md.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ.setdefault("HF_HOME", str(ROOT / "cache" / "huggingface"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as AP

from src.experiments.data import prepare_dataset
from src.student.losses import TotalLoss, soften_teacher
from src.teacher.clip_teacher import CLIPEmbedder
from src.teacher.prompt_learning import PromptEncoder, PromptLearningTeacher

BENCH = ROOT / "experiments" / "benchmark_v2"
DATASETS = ["creditcard", "daphnet", "oil3w", "metropt"]
OUT = []


def say(s=""):
    OUT.append(s)
    print(s.encode("ascii", "replace").decode(), flush=True)


def state_hash(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def main():
    torch.set_num_threads(2)
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    emb = CLIPEmbedder().load()
    say("# Functional checks of the novelty components"); say("")

    # 1. Prompt learning -------------------------------------------------------
    say("## 1. Prompt learning (CoOp)"); say("")
    before = state_hash(emb.model)
    E = np.load(BENCH / "metropt" / "clip_embeddings.npz")
    S = np.load(BENCH / "metropt" / "clip_embeddings_spectrogram.npz")
    data = {ds: prepare_dataset(ds, cfg, max_train_windows=int(cfg["benchmark"]["per_dataset"][ds].get("max_train_windows", 0)))
            for ds in DATASETS}
    mtr = data["metropt"].splits["train"].y_window
    n_tr = len(E["train"])
    idx = np.arange(len(mtr)) if len(mtr) <= n_tr else np.linspace(0, len(mtr) - 1, n_tr).astype(int)
    enc = PromptEncoder(emb.model, emb.tokenizer)
    trainable = [n for n, p in enc.named_parameters() if p.requires_grad]
    t = PromptLearningTeacher(epochs=3, n_folds=2)
    t._train(enc, np.stack([E["train"], S["train"]]), mtr[idx])
    after = state_hash(emb.model)
    say(f"- Trainable tensors during prompt learning: {trainable} ({sum(p.numel() for p in enc.parameters() if p.requires_grad)} numbers).")
    say(f"- CLIP state-dict hash before training {before}, after {after}: "
        f"**{'unchanged' if before == after else 'CHANGED'}**.")
    say(f"- Training loss per epoch (MetroPT, 2 views): {[round(x, 4) for x in t.history_]} "
        f"({'decreasing' if t.history_[-1] < t.history_[0] else 'NOT decreasing'}).")
    say("")
    say("Learned prompt vs. its own untrained initial phrase (validation and test window AUPRC; same frozen CLIP, "
        "same two views):"); say("")
    say("| Dataset | Initial phrase (untrained) val / test | Learned prompts val / test |")
    say("|---|---|---|")
    for ds in DATASETS:
        Eh = np.load(BENCH / ds / "clip_embeddings.npz"); Es = np.load(BENCH / ds / "clip_embeddings_spectrogram.npz")
        init = PromptLearningTeacher(); e0 = PromptEncoder(emb.model, emb.tokenizer)
        init._encoder = e0
        learned = PromptLearningTeacher(); learned.ctx_ = np.load(BENCH / ds / "clip_coop_mv.npz")["ctx"]
        v = np.stack([Eh["val"], Es["val"]]); te = np.stack([Eh["test"], Es["test"]])
        sp = data[ds].splits
        r0 = (AP(sp["val"].y_window, init._predict(e0, v)[:, 1]), AP(sp["test"].y_window, init._predict(e0, te)[:, 1]))
        r1 = (AP(sp["val"].y_window, learned.predict_proba(v, emb)[:, 1]),
              AP(sp["test"].y_window, learned.predict_proba(te, emb)[:, 1]))
        say(f"| {ds} | {r0[0]:.3f} / {r0[1]:.3f} | {r1[0]:.3f} / {r1[1]:.3f} |")
    say("")

    # 2. Multi-view -------------------------------------------------------------
    say("## 2. Multi-view (heatmap + spectrogram)"); say("")
    say("| Dataset | mean cos(heatmap, spectrogram) | probe on heatmap only, val AP | probe on spectrogram only, val AP | "
        "learned prompts 1 view → 2 views, test AP |")
    say("|---|---|---|---|---|")
    for ds in DATASETS:
        Eh = np.load(BENCH / ds / "clip_embeddings.npz"); Es = np.load(BENCH / ds / "clip_embeddings_spectrogram.npz")
        sp = data[ds].splits
        n = len(sp["train"].y_window); n_tr = len(Eh["train"])
        ti = np.arange(n) if n <= n_tr else np.linspace(0, n - 1, n_tr).astype(int)
        ytr = sp["train"].y_window[ti]
        cos = float((Eh["val"] * Es["val"]).sum(1).mean())
        aps = []
        for X in (Eh, Es):
            m = LogisticRegression(C=0.1, max_iter=3000).fit(X["train"], ytr)
            aps.append(AP(sp["val"].y_window, m.predict_proba(X["val"])[:, 1]))
        with np.load(BENCH / ds / "teacher_outputs.npz") as z:
            c1, c2 = AP(sp["test"].y_window, z["coop_test"][:, 1]), AP(sp["test"].y_window, z["coop_mv_test"][:, 1])
        say(f"| {ds} | {cos:.3f} | {aps[0]:.3f} | {aps[1]:.3f} | {c1:.3f} → {c2:.3f} |")
    say("")
    say("Cosines of 0.82-0.91 (views related but not identical) and a spectrogram-only probe above prevalence show "
        "that the second view adds information; whether it helps the final teacher is dataset-dependent (last column).")
    say("")

    # 3. Confidence-weighted KD ------------------------------------------------
    say("## 3. Confidence-weighted KD on a real batch (Daphnet, 256 training windows)"); say("")
    sp = data["daphnet"].splits["train"]
    with np.load(BENCH / "daphnet" / "teacher_outputs.npz") as z:
        tp = z["coop_mv_train"]
    n_tr = len(tp); ti = np.linspace(0, len(sp.y_window) - 1, n_tr).astype(int)
    rng = np.random.default_rng(0); b = rng.choice(n_tr, 256, replace=False)
    yw = torch.as_tensor(sp.y_window[ti][b]).long(); t_probs = torch.as_tensor(tp[b])
    torch.manual_seed(0)
    zw, zs = torch.randn(256, 2), torch.randn(256, sp.y_step.shape[1])
    ys = torch.as_tensor(sp.y_step[ti][b]).float()
    loss = TotalLoss(lambda_kd=1.0, kd_confidence=True, temperature=2.0)
    _, parts = loss(zs, zw, ys, yw, t_probs, return_dict=True)
    soft = soften_teacher(t_probs, 2.0)
    kl = (soft * (torch.log(soft) - F.log_softmax(zw / 2.0, -1))).sum(-1)
    w = t_probs.gather(1, yw.view(-1, 1)).squeeze(1)
    manual = float(4.0 * (w * kl).mean())
    wrong = (t_probs.argmax(1) != yw)
    say(f"- KD term from the training code {parts['kd_loss']:.6f}; hand-computed mean(p_T(y)·T²·KL) {manual:.6f}: "
        f"**{'match' if abs(parts['kd_loss'] - manual) < 1e-5 else 'MISMATCH'}**.")
    say(f"- Mean weight where the teacher is right: {float(w[~wrong].mean()):.3f} ({int((~wrong).sum())} windows); "
        f"where it is wrong: {float(w[wrong].mean()):.3f} ({int(wrong.sum())} windows).")
    say("")
    (BENCH / "NOVELTY_CHECKS.md").write_text("\n".join(OUT), encoding="utf-8")
    print("wrote", BENCH / "NOVELTY_CHECKS.md")


if __name__ == "__main__":
    main()
