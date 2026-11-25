"""
Trainer module for IE643 VLM Teacher-RNN Student Knowledge Distillation.

Supports three training modes:
- 'teacher_only': No student training, evaluates teacher predictions directly.
- 'student_baseline': Trains student using ground-truth labels only (lambda_kd = 0).
- 'student_kd': Trains student using both ground-truth labels and teacher soft targets.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import average_precision_score

from src.student.losses import TotalLoss

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Stores metrics and history from a training run."""

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    loc_losses: list[float] = field(default_factory=list)
    window_losses: list[float] = field(default_factory=list)
    kd_losses: list[float] = field(default_factory=list)
    feat_losses: list[float] = field(default_factory=list)
    val_auprc: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    training_time_seconds: float = 0.0

    def to_dict(self) -> dict:
        """Serializes training result to a JSON-compatible dictionary."""
        return {
            "train_losses": [float(x) for x in self.train_losses],
            "val_losses": [float(x) for x in self.val_losses],
            "loc_losses": [float(x) for x in self.loc_losses],
            "window_losses": [float(x) for x in self.window_losses],
            "kd_losses": [float(x) for x in self.kd_losses],
            "feat_losses": [float(x) for x in self.feat_losses],
            "val_auprc": [float(x) for x in self.val_auprc],
            "best_epoch": int(self.best_epoch),
            "best_val_loss": float(self.best_val_loss) if self.best_val_loss != float("inf") else 0.0,
            "training_time_seconds": float(self.training_time_seconds),
        }

    @classmethod
    def from_dict(cls, data: dict) -> TrainingResult:
        """Constructs TrainingResult from a dictionary."""
        return cls(
            train_losses=data.get("train_losses", []),
            val_losses=data.get("val_losses", []),
            loc_losses=data.get("loc_losses", []),
            window_losses=data.get("window_losses", []),
            kd_losses=data.get("kd_losses", []),
            feat_losses=data.get("feat_losses", []),
            val_auprc=data.get("val_auprc", []),
            best_epoch=data.get("best_epoch", 0),
            best_val_loss=data.get("best_val_loss", float("inf")),
            training_time_seconds=data.get("training_time_seconds", 0.0),
        )


class DistillationLoss(nn.Module):
    """
    Combined objective for joint window classification, step localization and KD.

        L = L_step + beta * L_window + lambda_kd * L_KD + lambda_feat * L_feat

    See ``src/student/losses.py`` for the definition of every term and a worked
    numeric example. The forward pass returns ``(total, step, window, kd)`` for
    backwards compatibility; the feature-distillation term is exposed as
    ``self.last_feat_loss``.
    """

    def __init__(
        self,
        temperature: float = 2.0,
        beta: float = 1.0,
        lambda_kd: float = 0.5,
        pos_weight: float = 10.0,
        lambda_feat: float = 0.0,
        kd_confidence: bool = False,
    ) -> None:
        super().__init__()
        self.temperature = float(temperature)
        self.beta = float(beta)
        self.lambda_kd = float(lambda_kd)
        self.lambda_feat = float(lambda_feat)
        self.pos_weight = float(pos_weight)
        self.objective = TotalLoss(
            beta=self.beta,
            lambda_kd=self.lambda_kd,
            pos_weight=self.pos_weight,
            temperature=self.temperature,
            lambda_feat=self.lambda_feat,
            kd_confidence=kd_confidence,
        )
        self.last_feat_loss = torch.tensor(0.0)

    def forward(
        self,
        window_logits: torch.Tensor,
        step_logits: torch.Tensor,
        y_window: torch.Tensor,
        y_step: torch.Tensor,
        teacher_probs: Optional[torch.Tensor] = None,
        teacher_mask: Optional[torch.Tensor] = None,
        student_embedding: Optional[torch.Tensor] = None,
        teacher_embedding: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if window_logits.ndim == 1 or (window_logits.ndim == 2 and window_logits.shape[-1] == 1):
            window_logits = torch.cat([-window_logits.view(-1, 1), window_logits.view(-1, 1)], dim=-1)
        if step_logits.ndim == 3 and step_logits.shape[-1] == 1:
            step_logits = step_logits.squeeze(-1)
        if teacher_probs is not None:
            teacher_probs = teacher_probs.to(window_logits.device, dtype=torch.float32)
            if teacher_probs.ndim == 2 and teacher_probs.shape[-1] == 1:
                teacher_probs = teacher_probs.view(-1)
            if teacher_mask is not None and len(teacher_mask) != len(teacher_probs):
                raise ValueError(
                    f"teacher_mask has {len(teacher_mask)} rows, but teacher_probs has {len(teacher_probs)}"
                )
        total, parts = self.objective(
            step_logits=step_logits,
            window_logits=window_logits,
            y_step=y_step.to(window_logits.device),
            y_window=y_window.to(window_logits.device),
            teacher_probs=teacher_probs,
            return_dict=True,
            teacher_mask=teacher_mask,
            student_embedding=student_embedding,
            teacher_embedding=teacher_embedding,
        )
        dev = window_logits.device
        self.last_feat_loss = torch.tensor(parts["feat_loss"], device=dev)
        return (
            total,
            torch.tensor(parts["loc_loss"], device=dev),
            torch.tensor(parts["win_loss"], device=dev),
            torch.tensor(parts["kd_loss"], device=dev),
        )


class Trainer:
    """
    Manages student model training, validation, early stopping, and predictions.
    """

    def __init__(
        self,
        model: Optional[nn.Module],
        loss_fn: Optional[Union[nn.Module, DistillationLoss]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[Union[str, torch.device]] = None,
        config: Optional[dict] = None,
    ) -> None:
        self.config = config or {}

        # Resolve compute device
        if device is None:
            dev_cfg = self.config.get("device", "auto")
            if dev_cfg == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(dev_cfg)
        elif isinstance(device, str):
            if device == "auto":
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)
        else:
            self.device = device

        self.model = model
        if self.model is not None:
            self.model.to(self.device)

        # Determine training mode
        self.mode = self.config.get("mode", self.config.get("training_mode", "student_kd"))
        if self.mode not in ("teacher_only", "student_baseline", "student_kd"):
            logger.warning("Unrecognized mode '%s', defaulting to 'student_kd'", self.mode)
            self.mode = "student_kd"

        # Distillation configuration
        distill_cfg = self.config.get("distillation", {})
        self.temperature = float(distill_cfg.get("temperature", 2.0))
        self.beta = float(distill_cfg.get("beta", 1.0))
        self.pos_weight = float(distill_cfg.get("pos_weight", 10.0))

        # Lambda KD is zeroed out in student_baseline mode
        if self.mode == "student_baseline":
            self.lambda_kd = 0.0
            self.lambda_feat = 0.0
        else:
            self.lambda_kd = float(distill_cfg.get("lambda_kd", 0.5))
            self.lambda_feat = float(distill_cfg.get("lambda_feat", 0.0))

        # Training parameters
        train_cfg = self.config.get("training", {})
        self.epochs = int(train_cfg.get("epochs", self.config.get("epochs", 30)))
        self.batch_size = int(train_cfg.get("batch_size", self.config.get("batch_size", 32)))
        self.learning_rate = float(train_cfg.get("learning_rate", self.config.get("learning_rate", 1e-3)))
        self.weight_decay = float(train_cfg.get("weight_decay", 1e-4))
        self.patience = int(train_cfg.get("early_stopping_patience", 5))
        self.gradient_clip = float(train_cfg.get("gradient_clip", 1.0))
        # "val_loss" (default) or "val_auprc": mean of validation window AUPRC and
        # step AUPRC, which tracks the reported metric more closely than the loss.
        self.selection_metric = str(train_cfg.get("selection_metric", "val_loss"))
        # Optional variance reduction: exponential moving average of the weights
        # (evaluated and checkpointed instead of the raw weights) and train-time
        # augmentation {"scale": s, "noise": n}: per-window, per-channel gain
        # 1 + s*N(0,1) and additive N(0, n^2) noise. Off by default.
        self.ema_decay = float(train_cfg.get("ema_decay", 0.0) or 0.0)
        self.augment = dict(train_cfg.get("augment") or {})
        self._ema: Optional[dict] = None

        # Loss function
        if loss_fn is not None:
            self.loss_fn = loss_fn
        else:
            self.loss_fn = DistillationLoss(
                temperature=self.temperature,
                beta=self.beta,
                lambda_kd=self.lambda_kd,
                pos_weight=self.pos_weight,
                lambda_feat=self.lambda_feat,
                kd_confidence=bool(distill_cfg.get("kd_confidence", False)),
            )

        # Optimizer
        if optimizer is not None:
            self.optimizer = optimizer
        elif self.model is not None and self.mode != "teacher_only":
            opt_name = str(train_cfg.get("optimizer", "adam")).lower()
            if opt_name == "adamw":
                self.optimizer = torch.optim.AdamW(
                    self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
                )
            elif opt_name == "sgd":
                self.optimizer = torch.optim.SGD(
                    self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, momentum=0.9
                )
            else:
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
                )
        else:
            self.optimizer = None

        # LR Scheduler
        if self.optimizer is not None:
            sched_patience = max(1, self.patience // 2)
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=sched_patience, min_lr=1e-6
            )
        else:
            self.scheduler = None

    def _unpack_outputs(self, outputs: Union[tuple, list, dict, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Normalizes student model outputs into (window_logits, step_logits).
        """
        if isinstance(outputs, dict):
            w = outputs.get("window_logits", outputs.get("logits_window", outputs.get("window", None)))
            s = outputs.get("step_logits", outputs.get("logits_step", outputs.get("step", None)))
            if w is None or s is None:
                raise ValueError("Model dictionary output must contain window_logits and step_logits")
            return w, s

        if isinstance(outputs, (tuple, list)):
            if len(outputs) >= 2:
                o0, o1 = outputs[0], outputs[1]
                # If first output is window_logits (usually shape (B, 2) or (B, 1))
                if o0.ndim <= 2 and (o0.shape[-1] <= 2 or (o1.ndim >= 2 and o1.shape[1] > o0.shape[-1])):
                    return o0, o1
                # If first output is step_logits (shape (B, L))
                return o1, o0
            raise ValueError(f"Expected at least 2 outputs from student model, got {len(outputs)}")

        raise TypeError(f"Unsupported model output type: {type(outputs)}")

    def _batch_teacher(self, batch: tuple) -> tuple:
        """Extract (teacher_probs, teacher_mask, teacher_embedding) from a batch."""
        t_probs = t_mask = t_emb = None
        if len(batch) >= 4:
            t_probs = batch[3].to(self.device, dtype=torch.float32)
        if len(batch) >= 5:
            t_mask = batch[4].to(self.device, dtype=torch.bool)
        if len(batch) >= 6:
            t_emb = batch[5].to(self.device, dtype=torch.float32)
        return t_probs, t_mask, t_emb

    def _ema_update(self) -> None:
        if self._ema is None:
            return
        d = self.ema_decay
        with torch.no_grad():
            for k, v in self.model.state_dict().items():
                if v.dtype.is_floating_point:
                    self._ema[k].mul_(d).add_(v.detach(), alpha=1.0 - d)
                else:
                    self._ema[k].copy_(v)

    def _step(self, batch: tuple, train: bool) -> dict[str, float]:
        X = batch[0].to(self.device, dtype=torch.float32)
        if train and self.augment:
            scale = float(self.augment.get("scale", 0.0))
            noise = float(self.augment.get("noise", 0.0))
            if scale > 0:
                X = X * (1.0 + scale * torch.randn(X.shape[0], 1, X.shape[2], device=X.device))
            if noise > 0:
                X = X + noise * torch.randn_like(X)
        y_step = batch[1].to(self.device, dtype=torch.float32)
        y_window = batch[2].to(self.device, dtype=torch.long)
        t_probs, t_mask, t_emb = self._batch_teacher(batch)
        use_teacher = self.mode == "student_kd" and train
        want_emb = (
            use_teacher and t_emb is not None and self.lambda_feat > 0
            and getattr(self.model, "embed_head", None) is not None
        )

        if want_emb:
            window_logits, step_logits, s_emb = self.model(X, return_embedding=True)
        else:
            window_logits, step_logits = self._unpack_outputs(self.model(X))
            s_emb = None

        if isinstance(self.loss_fn, DistillationLoss):
            loss, loc, win, kd = self.loss_fn(
                window_logits=window_logits,
                step_logits=step_logits,
                y_window=y_window,
                y_step=y_step,
                teacher_probs=t_probs if use_teacher else None,
                teacher_mask=t_mask if use_teacher else None,
                student_embedding=s_emb,
                teacher_embedding=t_emb if want_emb else None,
            )
            feat = float(self.loss_fn.last_feat_loss)
        else:
            out = self.loss_fn(window_logits, step_logits, y_window, y_step, t_probs)
            loss, loc, win, kd = out if isinstance(out, (tuple, list)) else (out, 0.0, 0.0, 0.0)
            feat = 0.0

        if train:
            self.optimizer.zero_grad()
            loss.backward()
            if self.gradient_clip > 0.0:
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)
            self.optimizer.step()
            self._ema_update()

        def as_float(v):
            return float(v.item()) if isinstance(v, torch.Tensor) else float(v)

        return {"total": as_float(loss), "loc": as_float(loc), "window": as_float(win),
                "kd": as_float(kd), "feat": feat}

    def _run_epoch(self, loader: DataLoader, train: bool) -> dict[str, float]:
        if self.model is None or (train and self.optimizer is None):
            return {"total": 0.0, "loc": 0.0, "window": 0.0, "kd": 0.0, "feat": 0.0}
        self.model.train(train)
        sums = {"total": 0.0, "loc": 0.0, "window": 0.0, "kd": 0.0, "feat": 0.0}
        n = 0
        with torch.set_grad_enabled(train):
            for batch in loader:
                for k, v in self._step(batch, train).items():
                    sums[k] += v
                n += 1
        return {k: v / max(1, n) for k, v in sums.items()}

    def train_epoch(self, train_loader: DataLoader, teacher_probs_dict: Optional[dict] = None) -> dict[str, float]:
        """Run one training epoch and return mean loss components."""
        return self._run_epoch(train_loader, train=True)

    def validate(self, val_loader: DataLoader, teacher_probs_dict: Optional[dict] = None) -> dict[str, float]:
        """Run one validation pass (supervised terms only) and return mean losses."""
        return self._run_epoch(val_loader, train=False)

    def _validation_auprc(self, X_val: np.ndarray, y_step_val: np.ndarray, y_window_val: np.ndarray) -> float:
        """Mean of window-level and step-level validation AUPRC (threshold-free)."""
        window_proba, step_scores = self.get_predictions(X_val, batch_size=256)
        values = []
        yw = np.asarray(y_window_val).astype(int).ravel()
        if 0 < yw.sum() < len(yw):
            values.append(average_precision_score(yw, window_proba[:, 1]))
        ys = np.asarray(y_step_val).astype(int).ravel()
        if 0 < ys.sum() < len(ys):
            values.append(average_precision_score(ys, step_scores.ravel()))
        return float(np.mean(values)) if values else 0.0

    def _create_loader(
        self,
        X: np.ndarray,
        y_step: np.ndarray,
        y_window: np.ndarray,
        teacher_probs: Optional[np.ndarray],
        teacher_mask: Optional[np.ndarray] = None,
        shuffle: bool = True,
        teacher_embeddings: Optional[np.ndarray] = None,
    ) -> DataLoader:
        """Build a DataLoader with layout (X, y_step, y_window[, t_probs, t_mask[, t_emb]])."""
        tensors = [
            torch.as_tensor(np.asarray(X), dtype=torch.float32),
            torch.as_tensor(np.asarray(y_step), dtype=torch.float32),
            torch.as_tensor(np.asarray(y_window), dtype=torch.long),
        ]
        if teacher_probs is not None or teacher_embeddings is not None:
            n = len(X)
            if teacher_probs is None:
                teacher_probs = np.tile(np.array([[0.5, 0.5]], dtype=np.float32), (n, 1))
            teacher_probs = np.asarray(teacher_probs, dtype=np.float32)
            if teacher_probs.ndim == 1:
                teacher_probs = np.stack([1.0 - teacher_probs, teacher_probs], axis=-1)
            if teacher_mask is None:
                teacher_mask = np.ones(len(teacher_probs), dtype=bool)
            teacher_mask = np.asarray(teacher_mask, dtype=bool).reshape(-1)
            if len(teacher_mask) != len(teacher_probs):
                raise ValueError(
                    f"teacher_mask has {len(teacher_mask)} rows, but teacher_probs has {len(teacher_probs)}"
                )
            tensors.append(torch.as_tensor(teacher_probs, dtype=torch.float32))
            tensors.append(torch.as_tensor(teacher_mask, dtype=torch.bool))
            if teacher_embeddings is not None:
                tensors.append(torch.as_tensor(np.asarray(teacher_embeddings), dtype=torch.float32))
        return DataLoader(TensorDataset(*tensors), batch_size=self.batch_size, shuffle=shuffle, drop_last=False)

    def fit(
        self,
        X_train: np.ndarray,
        y_step_train: np.ndarray,
        y_window_train: np.ndarray,
        teacher_probs_train: Optional[np.ndarray],
        X_val: np.ndarray,
        y_step_val: np.ndarray,
        y_window_val: np.ndarray,
        teacher_probs_val: Optional[np.ndarray],
        progress_callback: Optional[Callable[[int, int, float, float], None]] = None,
        teacher_mask_train: Optional[np.ndarray] = None,
        teacher_mask_val: Optional[np.ndarray] = None,
        teacher_embeddings_train: Optional[np.ndarray] = None,
        epoch_callback: Optional[Callable[[int, "TrainingResult"], None]] = None,
    ) -> TrainingResult:
        """Train with early stopping and restore the best epoch's weights.

        Teacher targets (probabilities and optional CLIP embeddings) are used
        only for training windows where ``teacher_mask_train`` is True. The
        validation loss contains the supervised terms only, so baseline and KD
        runs are selected with the same criterion.
        """
        logger.info(
            "Starting training in mode '%s': %d train windows, %d val windows, max epochs = %d",
            self.mode, len(X_train), len(X_val), self.epochs,
        )
        start_time = time.time()
        result = TrainingResult()

        if self.mode == "teacher_only":
            if progress_callback:
                progress_callback(1, 1, 0.0, 0.0)
            result.training_time_seconds = time.time() - start_time
            result.best_val_loss = 0.0
            return result

        use_teacher = self.mode == "student_kd"
        train_loader = self._create_loader(
            X_train, y_step_train, y_window_train,
            teacher_probs_train if use_teacher else None,
            teacher_mask=teacher_mask_train if use_teacher else None,
            shuffle=True,
            teacher_embeddings=(
                teacher_embeddings_train if use_teacher and self.lambda_feat > 0 else None
            ),
        )
        val_loader = self._create_loader(X_val, y_step_val, y_window_val, None, shuffle=False)

        use_auprc = self.selection_metric == "val_auprc"
        best_score = -float("inf")
        best_epoch = 0
        best_state = None
        patience_counter = 0

        if self.ema_decay > 0:
            self._ema = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

        for epoch in range(1, self.epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            live_state = None
            if self._ema is not None:
                # Validate (and checkpoint) the averaged weights.
                live_state = copy.deepcopy(self.model.state_dict())
                self.model.load_state_dict(self._ema)
            val_metrics = self.validate(val_loader)
            train_loss, val_loss = train_metrics["total"], val_metrics["total"]

            result.train_losses.append(train_loss)
            result.val_losses.append(val_loss)
            result.loc_losses.append(train_metrics["loc"])
            result.window_losses.append(train_metrics["window"])
            result.kd_losses.append(train_metrics["kd"])
            result.feat_losses.append(train_metrics["feat"])

            val_auprc = (
                self._validation_auprc(X_val, y_step_val, y_window_val) if use_auprc else float("nan")
            )
            result.val_auprc.append(val_auprc)
            score = val_auprc if use_auprc else -val_loss

            if self.scheduler is not None:
                self.scheduler.step(-score)

            if score > best_score + 1e-5:
                best_score = score
                best_epoch = epoch
                result.best_val_loss = val_loss
                patience_counter = 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
            if live_state is not None:
                self.model.load_state_dict(live_state)

            logger.debug(
                "Epoch %02d | train %.4f (loc %.4f win %.4f kd %.4f feat %.4f) | val %.4f | val_auprc %.4f",
                epoch, train_loss, train_metrics["loc"], train_metrics["window"],
                train_metrics["kd"], train_metrics["feat"], val_loss, val_auprc,
            )
            if progress_callback is not None:
                try:
                    progress_callback(epoch, self.epochs, train_loss, val_loss)
                except Exception as cb_err:
                    logger.warning("Error in progress_callback: %s", cb_err)
            if epoch_callback is not None:
                # Full history so far (all loss components, validation AUPRC,
                # best epoch) so a UI can redraw the training curves live.
                result.best_epoch = best_epoch
                try:
                    epoch_callback(epoch, result)
                except Exception as cb_err:
                    logger.warning("Error in epoch_callback: %s", cb_err)
            if patience_counter >= self.patience:
                logger.info("Early stopping at epoch %d", epoch)
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        result.best_epoch = best_epoch
        result.training_time_seconds = time.time() - start_time
        return result

    def get_predictions(
        self,
        X: np.ndarray,
        batch_size: int = 64,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes window anomaly probabilities and step-level anomaly scores.

        Args:
            X: Input window array, shape (N, window_length, input_dim)
            batch_size: Batch size for forward passes

        Returns:
            Tuple of:
                - window_proba: Shape (N, 2) where col 0 is normal prob, col 1 is anomaly prob
                - step_scores: Shape (N, window_length) step-level anomaly probabilities
        """
        if self.model is None:
            n_samples = len(X)
            w_len = X.shape[1] if X.ndim > 1 else 128
            logger.warning("No model loaded in Trainer. Returning neutral zeros.")
            neutral_window = np.column_stack([np.ones(n_samples), np.zeros(n_samples)])
            neutral_steps = np.zeros((n_samples, w_len), dtype=np.float32)
            return neutral_window, neutral_steps

        self.model.eval()
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        window_probs_list: list[np.ndarray] = []
        step_scores_list: list[np.ndarray] = []

        with torch.no_grad():
            for (batch_x,) in loader:
                batch_x = batch_x.to(self.device)
                outputs = self.model(batch_x)
                w_logits, s_logits = self._unpack_outputs(outputs)

                # Window probabilities (N_batch, 2)
                if w_logits.ndim == 1:
                    p1 = torch.sigmoid(w_logits).view(-1, 1)
                    w_prob = torch.cat([1.0 - p1, p1], dim=-1)
                elif w_logits.shape[-1] == 1:
                    p1 = torch.sigmoid(w_logits)
                    w_prob = torch.cat([1.0 - p1, p1], dim=-1)
                else:
                    w_prob = F.softmax(w_logits, dim=-1)

                # Step scores (N_batch, L)
                if s_logits.ndim == 3 and s_logits.shape[-1] == 1:
                    s_logits = s_logits.squeeze(-1)
                s_prob = torch.sigmoid(s_logits)

                window_probs_list.append(w_prob.cpu().numpy())
                step_scores_list.append(s_prob.cpu().numpy())

        all_window_proba = np.concatenate(window_probs_list, axis=0)
        all_step_scores = np.concatenate(step_scores_list, axis=0)

        return all_window_proba, all_step_scores
