"""Loss functions for the GRU student and VLM knowledge distillation.

The full training objective for one mini-batch is

    L = L_step + beta * L_window + lambda_kd * L_KD + lambda_feat * L_feat

    L_step   weighted BCE(step_logits, y_step)              -> step head, GT labels
    L_window CE(window_logits, y_window)                     -> window head, GT labels
    L_KD     T^2 * KL(softmax(log p_teacher / T) || softmax(z_window / T))
                                                             -> window head, teacher soft labels
    L_feat   1 - cos(g(h_pooled), e_CLIP)                    -> embedding head, CLIP image embedding

L_KD and L_feat are averaged only over windows that the frozen teacher scored
(``teacher_mask``). With ``kd_confidence=True`` each window's KD term is
multiplied by the teacher's (unsoftened) probability for the ground-truth class,
w_i = p_teacher(y_i): windows the teacher gets wrong are distilled less, and a
teacher that is often wrong contributes less overall. With lambda_kd = lambda_feat = 0 this is the supervised
baseline; with beta = 0 the window head learns from the teacher alone.

Worked example (one window, T = 2, beta = 1, lambda_kd = 1, lambda_feat = 0.5):
    ground truth           y_window = 1, y_step = [0, 0, 1, 1]
    teacher probabilities  p_teacher = [0.2, 0.8]  -> softened [0.333, 0.667]
    student window logits  z = [0.1, 0.9]          -> softened [0.401, 0.599]
    L_window = -log softmax(z)[1]           = 0.371
    L_KD     = 4 * KL([.333,.667] || [.401,.599]) = 4 * 0.0098 = 0.039
    L_step   = mean weighted BCE over the 4 steps (positives weighted by pos_weight)
    L_feat   = 1 - cosine(student embedding, CLIP embedding)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalizationLoss(nn.Module):
    """Weighted binary cross-entropy for timestep anomaly localization."""

    def __init__(self, pos_weight: float = 10.0) -> None:
        super().__init__()
        self.pos_weight_val = float(pos_weight)
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))

    def forward(self, step_logits: torch.Tensor, y_step: torch.Tensor) -> torch.Tensor:
        if step_logits.ndim == 3 and step_logits.shape[-1] == 1:
            step_logits = step_logits.squeeze(-1)
        y_true = y_step.to(step_logits.device, dtype=torch.float32)
        return F.binary_cross_entropy_with_logits(
            step_logits, y_true, pos_weight=self.pos_weight.to(step_logits.device)
        )


class WindowClassificationLoss(nn.Module):
    """Cross-entropy for window-level anomaly classification."""

    def __init__(self) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, window_logits: torch.Tensor, y_window: torch.Tensor) -> torch.Tensor:
        return self.ce(window_logits, y_window.to(window_logits.device, dtype=torch.long))


def soften_teacher(teacher_probs: torch.Tensor, temperature: float) -> torch.Tensor:
    """Soften teacher probabilities (or logits) with temperature T.

    Probabilities are mapped back to log space (logits up to a constant) and
    divided by T, which equals ``softmax(teacher_logits / T)``.
    """
    t = teacher_probs.detach().to(dtype=torch.float32)
    if t.ndim == 1:
        t = torch.stack([1.0 - t, t], dim=-1)
    row_sums = t.sum(dim=-1)
    is_prob = bool(torch.all(t >= 0).item()) and torch.allclose(
        row_sums, torch.ones_like(row_sums), atol=1e-3
    )
    logits = torch.log(t.clamp_min(1e-8)) if is_prob else t
    return F.softmax(logits / temperature, dim=-1)


class KnowledgeDistillationLoss(nn.Module):
    """Hinton KD: ``T^2 * KL(p_teacher(T) || p_student(T))`` (batch mean)."""

    def __init__(self, temperature: float = 2.0) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, student_logits: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
        T = self.temperature
        if T <= 0:
            raise ValueError(f"temperature must be positive, got {T}")
        student_log_p = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft_p = soften_teacher(teacher_probs.to(student_logits.device), T)
        return (T * T) * self.kl_div(student_log_p, teacher_soft_p)


class FeatureDistillationLoss(nn.Module):
    """Cosine embedding distillation: ``1 - cos(student_proj, teacher_embedding)``."""

    def forward(self, student_embedding: torch.Tensor, teacher_embedding: torch.Tensor) -> torch.Tensor:
        s = F.normalize(student_embedding, dim=-1)
        t = F.normalize(teacher_embedding.detach().to(s.device, dtype=s.dtype), dim=-1)
        return (1.0 - (s * t).sum(dim=-1)).mean()


class TotalLoss(nn.Module):
    """``L_step + beta*L_window + lambda_kd*L_KD + lambda_feat*L_feat`` (see module docstring)."""

    def __init__(
        self,
        beta: float = 1.0,
        lambda_kd: float = 0.5,
        pos_weight: float = 10.0,
        temperature: float = 2.0,
        lambda_feat: float = 0.0,
        kd_confidence: bool = False,
    ) -> None:
        super().__init__()
        self.kd_confidence = bool(kd_confidence)
        self.beta = float(beta)
        self.lambda_kd = float(lambda_kd)
        self.lambda_feat = float(lambda_feat)
        self.loc_loss_fn = LocalizationLoss(pos_weight=pos_weight)
        self.win_loss_fn = WindowClassificationLoss()
        self.kd_loss_fn = KnowledgeDistillationLoss(temperature=temperature)
        self.feat_loss_fn = FeatureDistillationLoss()

    def forward(
        self,
        step_logits: torch.Tensor,
        window_logits: torch.Tensor,
        y_step: torch.Tensor,
        y_window: torch.Tensor,
        teacher_probs: Optional[torch.Tensor] = None,
        return_dict: bool = False,
        teacher_mask: Optional[torch.Tensor] = None,
        student_embedding: Optional[torch.Tensor] = None,
        teacher_embedding: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        device = window_logits.device
        l_loc = self.loc_loss_fn(step_logits, y_step)
        l_win = self.win_loss_fn(window_logits, y_window)
        total = l_loc + self.beta * l_win

        rows = None
        if teacher_mask is not None:
            rows = torch.as_tensor(teacher_mask, device=device).bool().reshape(-1)

        l_kd = torch.tensor(0.0, device=device)
        if teacher_probs is not None and self.lambda_kd > 0:
            t = teacher_probs.to(device)
            z = window_logits
            yw = y_window.to(device).long()
            if rows is not None:
                t, z, yw = t[rows], z[rows], yw[rows]
            if len(z):
                if self.kd_confidence:
                    T = self.kd_loss_fn.temperature
                    t2 = t if t.ndim == 2 else torch.stack([1 - t, t], dim=-1)
                    soft_t = soften_teacher(t2, T)
                    log_s = F.log_softmax(z / T, dim=-1)
                    kl_rows = (soft_t * (torch.log(soft_t.clamp_min(1e-8)) - log_s)).sum(-1)
                    w = t2.detach().gather(1, yw.view(-1, 1)).squeeze(1)
                    l_kd = (T * T) * (w * kl_rows).mean()
                else:
                    l_kd = self.kd_loss_fn(z, t)
                total = total + self.lambda_kd * l_kd

        l_feat = torch.tensor(0.0, device=device)
        if student_embedding is not None and teacher_embedding is not None and self.lambda_feat > 0:
            s, e = student_embedding, teacher_embedding.to(device)
            if rows is not None:
                s, e = s[rows], e[rows]
            if len(s):
                l_feat = self.feat_loss_fn(s, e)
                total = total + self.lambda_feat * l_feat

        if return_dict:
            return total, {
                "total_loss": float(total.detach().item()),
                "loc_loss": float(l_loc.detach().item()),
                "win_loss": float(l_win.detach().item()),
                "kd_loss": float(l_kd.detach().item()),
                "feat_loss": float(l_feat.detach().item()),
            }
        return total
