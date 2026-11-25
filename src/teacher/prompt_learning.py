"""Prompt-learning teacher (CoOp-style) on top of frozen CLIP.

Zero-shot CLIP scores an image against hand-written prompts such as
"a plot of an anomalous time series". Context Optimization (CoOp; Zhou et al.,
IJCV 2022) instead *learns* the words of the prompt: the M context token
embeddings in

    [SOS] [v_1] [v_2] ... [v_M] <class name> . [EOS]

are free 512-d vectors, while every CLIP weight (image encoder, text encoder,
token embeddings, projections) stays frozen. Only M x 512 numbers are trained,
by cross-entropy between the training labels and

    p(class c | image) = softmax_c( s * cos(e_image, f_text(context, class c)) ),

with s = CLIP's own logit scale (100). The context starts from the embedding of
a natural phrase, so before training the teacher equals a zero-shot prompt.

Multi-view: when several image embeddings per window are given (heatmap and
spectrogram renderings), the class logits are averaged over the views, using
the same learned prompts.

Cross-fitting: like the probe, training windows are split into interleaved
blocks; each block's soft label comes from prompts learned on the other folds,
so the student never distils a prompt that saw that window's label.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.teacher.probe import _block_folds

INIT_PHRASE = "a heatmap of a multivariate sensor time series that looks"
CLASS_NAMES = ("normal", "anomalous")


class PromptEncoder(torch.nn.Module):
    """Frozen CLIP text encoder with M learnable context vectors."""

    def __init__(self, clip_model, tokenizer, init_phrase: str = INIT_PHRASE,
                 class_names: Sequence[str] = CLASS_NAMES) -> None:
        super().__init__()
        self.clip = clip_model
        for p in self.clip.parameters():
            p.requires_grad = False
        tm = clip_model.text_model
        init_ids = tokenizer(init_phrase, add_special_tokens=False)["input_ids"]
        self.n_ctx = len(init_ids)
        with torch.no_grad():
            init = tm.embeddings.token_embedding(torch.tensor(init_ids))
        self.ctx = torch.nn.Parameter(init.clone())                       # (M, D) - the only trainable tensor
        placeholder = " ".join(["X"] * self.n_ctx)
        tok = tokenizer([f"{placeholder} {c}." for c in class_names], padding=True, return_tensors="pt")
        self.register_buffer("ids", tok["input_ids"], persistent=False)   # (C, S)
        eos = tm.config.eos_token_id
        self.eos_pos = (self.ids.argmax(-1) if eos == 2 else (self.ids == eos).int().argmax(-1))

    def text_features(self) -> torch.Tensor:
        """Unit-norm text features (C, 512) for the current context."""
        from transformers.masking_utils import create_causal_mask

        tm = self.clip.text_model
        emb = tm.embeddings.token_embedding(self.ids)                     # (C, S, D)
        ctx = self.ctx.unsqueeze(0).expand(emb.shape[0], -1, -1)
        emb = torch.cat([emb[:, :1], ctx, emb[:, 1 + self.n_ctx:]], dim=1)
        hidden = tm.embeddings(inputs_embeds=emb)                          # + position embeddings
        mask = create_causal_mask(config=tm.config, inputs_embeds=hidden, attention_mask=None,
                                  past_key_values=None)
        out = tm.encoder(inputs_embeds=hidden, attention_mask=mask, is_causal=True)
        last = tm.final_layer_norm(out.last_hidden_state)
        pooled = last[torch.arange(last.shape[0]), self.eos_pos]
        return F.normalize(self.clip.text_projection(pooled), dim=-1)

    def logits(self, views: torch.Tensor) -> torch.Tensor:
        """views: (V, N, 512) unit image embeddings -> (N, C) view-averaged logits."""
        t = self.text_features()
        scale = self.clip.logit_scale.exp()
        return (scale * torch.einsum("vnd,cd->vnc", views, t)).mean(0)


@dataclass
class PromptLearningTeacher:
    """CoOp teacher: learns prompt context on frozen CLIP; cross-fitted soft labels."""

    epochs: int = 15
    lr: float = 2e-3
    batch_size: int = 256
    n_folds: int = 5
    block_size: int = 32
    seed: int = 0
    ctx_: Optional[np.ndarray] = None
    history_: List[float] = field(default_factory=list)
    _encoder: object = None

    def _new_encoder(self, embedder) -> PromptEncoder:
        embedder.load()
        return PromptEncoder(embedder.model, embedder.tokenizer)

    def _train(self, enc: PromptEncoder, views: np.ndarray, y: np.ndarray) -> PromptEncoder:
        torch.manual_seed(self.seed)
        init = enc.ctx.detach().clone()
        enc.ctx.data.copy_(init)
        opt = torch.optim.Adam([enc.ctx], lr=self.lr)
        V = torch.as_tensor(views, dtype=torch.float32)
        Y = torch.as_tensor(y, dtype=torch.long)
        n = V.shape[1]
        gen = torch.Generator().manual_seed(self.seed)
        losses = []
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=gen)
            total = 0.0
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                loss = F.cross_entropy(enc.logits(V[:, idx]), Y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss) * len(idx)
            losses.append(total / n)
        self.history_ = losses
        return enc

    @torch.no_grad()
    def _predict(self, enc: PromptEncoder, views: np.ndarray) -> np.ndarray:
        V = torch.as_tensor(views, dtype=torch.float32)
        return torch.softmax(enc.logits(V), dim=-1).numpy().astype(np.float32)

    def fit(self, embedder, views: np.ndarray, y: np.ndarray) -> np.ndarray:
        """views: (V, N, 512). Returns out-of-fold (N, 2) probabilities and fits the
        final prompts on all windows."""
        y = np.asarray(y).astype(int).ravel()
        enc = self._new_encoder(embedder)
        init = enc.ctx.detach().clone()
        oof = np.zeros((len(y), 2), np.float32)
        for idx in _block_folds(len(y), self.n_folds, self.block_size):
            mask = np.ones(len(y), bool)
            mask[idx] = False
            enc.ctx.data.copy_(init)
            if y[mask].min() == y[mask].max():
                oof[idx] = [1 - y.mean(), y.mean()]
                continue
            self._train(enc, views[:, mask], y[mask])
            oof[idx] = self._predict(enc, views[:, idx])
        enc.ctx.data.copy_(init)
        self._train(enc, views, y)
        self.ctx_ = enc.ctx.detach().numpy().copy()
        self._encoder = enc
        return oof

    def predict_proba(self, views: np.ndarray, embedder=None) -> np.ndarray:
        if self._encoder is None:
            enc = self._new_encoder(embedder)
            enc.ctx.data.copy_(torch.as_tensor(self.ctx_))
            self._encoder = enc
        return self._predict(self._encoder, views)

    def nearest_words(self, embedder, k: int = 1) -> List[str]:
        """Closest vocabulary token to each learned context vector (interpretability)."""
        emb = self._encoder.clip.text_model.embeddings.token_embedding.weight.detach()
        ctx = torch.as_tensor(self.ctx_)
        sims = F.normalize(ctx, dim=-1) @ F.normalize(emb, dim=-1).T
        ids = sims.topk(k, dim=-1).indices[:, 0].tolist()
        return [embedder.tokenizer.decode([i]).strip() for i in ids]

    def state(self) -> Dict[str, object]:
        return {"ctx": self.ctx_, "init_phrase": INIT_PHRASE, "class_names": list(CLASS_NAMES)}
