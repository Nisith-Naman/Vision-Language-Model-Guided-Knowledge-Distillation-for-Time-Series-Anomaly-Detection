"""
CLIP-based Vision-Language Model teacher implementation for knowledge distillation.

Encodes rendered time-series window images alongside nominal and anomaly prompts
to produce soft supervision signals (probabilities) for the student model.
"""

import os
import pickle
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from tqdm import tqdm

from src.teacher.base import BaseTeacher
from src.teacher.image_gen import batch_to_images

# Global warning constant regarding teacher calibration
WARNING: str = (
    "CLIP was not trained for time-series anomaly detection. "
    "Teacher scores are candidate signals, not calibrated anomaly probabilities."
)

try:
    from transformers import CLIPModel, CLIPProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    CLIPModel = None
    CLIPProcessor = None
    HAS_TRANSFORMERS = False


class CLIPTeacher(BaseTeacher):
    """
    CLIP teacher model for time-series anomaly detection.

    Computes cosine similarity between window images and contrastive prompts:
    - Normal prompt: e.g. 'A normal regular time series signal without anomalies'
    - Anomaly prompt: e.g. 'An anomalous irregular time series signal with spikes or distortions'
    """

    WARNING: str = WARNING
    CALIBRATION_WARNING: str = WARNING

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cpu",
    ) -> None:
        """
        Initialize CLIP teacher.

        Args:
            model_name: Hugging Face model identifier for pretrained CLIP.
            device: 'cpu', 'cuda', or 'auto'.
        """
        self.model_name = model_name

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model: Optional[Any] = None
        self.processor: Optional[Any] = None

    def load_model(self) -> None:
        """
        Load CLIPModel and CLIPProcessor from Hugging Face transformers.

        Puts the model in eval mode and freezes all parameters.
        """
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "Hugging Face transformers is required to run CLIPTeacher. "
                "Please install it using: pip install transformers"
            )

        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)

        # Set evaluation mode
        self.model.eval()

        # Freeze ALL parameters
        for p in self.model.parameters():
            p.requires_grad = False

        # Verify all parameters are strictly frozen
        assert not any(p.requires_grad for p in self.model.parameters()), (
            "Teacher model parameters were not properly frozen!"
        )

    def is_frozen(self) -> bool:
        """Verify whether the model is loaded and all parameters are frozen."""
        if self.model is None:
            return False
        return not any(p.requires_grad for p in self.model.parameters())

    def score_window(
        self,
        image: Image.Image,
        normal_prompt: str,
        anomaly_prompt: str,
    ) -> np.ndarray:
        """
        Score a single window image against the normal and anomaly prompts.

        Args:
            image: PIL Image representing the time-series window.
            normal_prompt: Nominal text prompt.
            anomaly_prompt: Anomalous text prompt.

        Returns:
            np.ndarray: Probability vector [p_normal, p_anomaly] summing to 1.0.
        """
        if self.model is None or self.processor is None:
            self.load_model()

        prompts = [normal_prompt, anomaly_prompt]
        inputs = self.processor(
            text=prompts,
            images=image,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # outputs.logits_per_image has shape (1, 2)
            logits = outputs.logits_per_image
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy().astype(np.float32)

        # Renormalize to ensure exact sum to 1.0
        prob_sum = probs.sum()
        if prob_sum > 0:
            probs = probs / prob_sum
        else:
            probs = np.array([0.5, 0.5], dtype=np.float32)

        return probs

    def score_batch(
        self,
        images: List[Image.Image],
        normal_prompt: str,
        anomaly_prompt: str,
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Score a list of window images against prompts in mini-batches.

        Args:
            images: List of PIL Images.
            normal_prompt: Nominal text prompt.
            anomaly_prompt: Anomalous text prompt.
            batch_size: Mini-batch size for processor and model inference.

        Returns:
            np.ndarray: (N, 2) array of [p_normal, p_anomaly] for each image.
        """
        if not images:
            return np.empty((0, 2), dtype=np.float32)

        if self.model is None or self.processor is None:
            self.load_model()

        prompts = [normal_prompt, anomaly_prompt]
        all_probs: List[np.ndarray] = []
        num_images = len(images)

        progress = tqdm(
            range(0, num_images, batch_size),
            desc="Scoring images with CLIP",
            unit="batch",
        )

        for start_idx in progress:
            batch_imgs = images[start_idx : start_idx + batch_size]
            inputs = self.processor(
                text=prompts,
                images=batch_imgs,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                # outputs.logits_per_image has shape (len(batch_imgs), 2)
                batch_logits = outputs.logits_per_image
                batch_probs = torch.softmax(batch_logits, dim=-1).cpu().numpy().astype(np.float32)

            row_sums = batch_probs.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            batch_probs = batch_probs / row_sums
            all_probs.append(batch_probs)

        return np.vstack(all_probs)

    def cache_and_score(
        self,
        windows: np.ndarray,
        feature_names: List[str],
        normal_prompt: str,
        anomaly_prompt: str,
        image_mode: str = "heatmap",
        image_size: Tuple[int, int] = (224, 224),
        cache_path: Optional[str] = None,
        max_windows: int = 5000,
    ) -> np.ndarray:
        """
        Score windows with caching support and a window budget cap.

        If a valid cache exists at cache_path, loads and returns cached scores.
        Otherwise, converts windows to images, scores them via CLIP, and saves
        the resulting scores to cache_path.

        Args:
            windows: 3D array of shape (N, L, F) containing time-series windows.
            feature_names: Feature names for the window channels.
            normal_prompt: Nominal text prompt.
            anomaly_prompt: Anomalous text prompt.
            image_mode: Visual representation ('heatmap' or 'lineplot').
            image_size: Image dimensions (width, height).
            cache_path: File path to save/load cached scores.
            max_windows: Maximum number of windows to score (budget cap to prevent OOM/slowdown).
                         If max_windows <= 0 or None, scores all windows.

        Returns:
            np.ndarray: (N, 2) array of [p_normal, p_anomaly] for each window scored.
        """
        # 1. Attempt to load from disk cache
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    cached = pickle.load(f)

                if isinstance(cached, dict) and "scores" in cached:
                    scores = cached["scores"]
                elif isinstance(cached, np.ndarray):
                    scores = cached
                elif isinstance(cached, tuple) and len(cached) > 0 and isinstance(cached[0], np.ndarray):
                    scores = cached[0]
                else:
                    scores = None

                if scores is not None and isinstance(scores, np.ndarray):
                    if max_windows is not None and max_windows > 0 and len(scores) > max_windows:
                        scores = scores[:max_windows]
                    return scores.astype(np.float32)

            except Exception as exc:
                warnings.warn(
                    f"Failed to load cache from '{cache_path}': {exc}. Recomputing scores."
                )

        # 2. Respect max_windows cap
        total_windows = len(windows)
        if max_windows is not None and max_windows > 0 and total_windows > max_windows:
            windows_to_score = windows[:max_windows]
        else:
            windows_to_score = windows

        # 3. Generate images
        images = batch_to_images(
            windows=windows_to_score,
            feature_names=feature_names,
            mode=image_mode,
            image_size=image_size,
        )

        # 4. Score images
        scores = self.score_batch(
            images=images,
            normal_prompt=normal_prompt,
            anomaly_prompt=anomaly_prompt,
        )

        # 5. Save to disk cache
        if cache_path:
            cache_dir = os.path.dirname(os.path.abspath(cache_path))
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)

            cache_data = {
                "scores": scores,
                "metadata": {
                    "num_windows": len(scores),
                    "model_name": self.model_name,
                    "normal_prompt": normal_prompt,
                    "anomaly_prompt": anomaly_prompt,
                    "image_mode": image_mode,
                    "image_size": image_size,
                },
            }
            with open(cache_path, "wb") as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        return scores

    def evaluate_on_labels(
        self,
        teacher_scores: np.ndarray,
        y_true: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Evaluate teacher anomaly scoring performance against ground-truth labels.

        Args:
            teacher_scores: (N, 2) array of [p_normal, p_anomaly] probabilities.
            y_true: (N,) binary labels (0 for normal, 1 for anomaly).

        Returns:
            Dict[str, Any] containing AUPRC, ROC-AUC, and calibration warning.
        """
        y_true_arr = np.asarray(y_true).astype(int).ravel()
        scores_arr = np.asarray(teacher_scores)

        if scores_arr.ndim == 2 and scores_arr.shape[1] == 2:
            p_anomaly = scores_arr[:, 1].astype(np.float64)
        else:
            p_anomaly = scores_arr.ravel().astype(np.float64)

        # Align lengths in case of mismatch
        min_len = min(len(y_true_arr), len(p_anomaly))
        y_true_arr = y_true_arr[:min_len]
        p_anomaly = p_anomaly[:min_len]

        results: Dict[str, Any] = {
            "warning": self.WARNING,
            "calibration_warning": self.WARNING,
        }

        unique_labels = np.unique(y_true_arr)
        if len(unique_labels) > 1:
            try:
                results["auprc"] = float(average_precision_score(y_true_arr, p_anomaly))
            except Exception:
                results["auprc"] = float("nan")

            try:
                results["roc_auc"] = float(roc_auc_score(y_true_arr, p_anomaly))
            except Exception:
                results["roc_auc"] = float("nan")
        else:
            results["auprc"] = float("nan")
            results["roc_auc"] = float("nan")

        return results


# ---------------------------------------------------------------------------
# Embedding-based interface (used by the benchmark and the linear-probe teacher)
# ---------------------------------------------------------------------------

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

NORMAL_PROMPTS = [
    "a plot of a normal, regular time series",
    "a smooth and stable sensor signal",
    "a chart of periodic, consistent measurements",
    "a time series without anomalies",
]
ANOMALY_PROMPTS = [
    "a plot of an anomalous, irregular time series",
    "a sensor signal with sudden spikes or a level shift",
    "a chart of erratic, noisy measurements with a fault",
    "a time series containing an anomaly",
]


def _pooled(output: Any) -> torch.Tensor:
    """Handle both tensor (transformers<5) and ModelOutput (>=5) feature returns."""
    if isinstance(output, torch.Tensor):
        return output
    pooled = getattr(output, "pooler_output", None)
    if pooled is None:
        raise TypeError(f"Unexpected CLIP feature output type: {type(output)}")
    return pooled


class CLIPEmbedder:
    """Frozen CLIP image/text encoder that returns L2-normalised embeddings.

    Images are rendered at 224x224, so CLIP's resize/crop is the identity and
    only the channel normalisation is applied (done here in NumPy).
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else (
            "cpu" if device == "auto" else device
        )
        self.model = None
        self.tokenizer = None

    def load(self) -> "CLIPEmbedder":
        if self.model is None:
            from transformers import CLIPModel as _CLIPModel, CLIPTokenizer

            self.model = _CLIPModel.from_pretrained(self.model_name).to(self.device).eval()
            for p in self.model.parameters():
                p.requires_grad = False
            self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
        return self

    def is_frozen(self) -> bool:
        return self.model is not None and not any(p.requires_grad for p in self.model.parameters())

    @torch.no_grad()
    def encode_images(self, images_uint8: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Encode ``(N, 224, 224, 3)`` uint8 images to ``(N, D)`` unit vectors."""
        self.load()
        out = []
        for start in range(0, len(images_uint8), batch_size):
            batch = images_uint8[start:start + batch_size].astype(np.float32) / 255.0
            batch = (batch - CLIP_MEAN) / CLIP_STD
            pixel_values = torch.from_numpy(batch.transpose(0, 3, 1, 2).copy()).to(self.device)
            feats = _pooled(self.model.get_image_features(pixel_values=pixel_values))
            out.append(torch.nn.functional.normalize(feats, dim=-1).cpu().numpy())
        return np.concatenate(out, axis=0).astype(np.float32) if out else np.empty((0, 512), np.float32)

    @torch.no_grad()
    def encode_texts(self, prompts: List[str]) -> np.ndarray:
        self.load()
        tokens = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)
        feats = _pooled(self.model.get_text_features(**tokens))
        return torch.nn.functional.normalize(feats, dim=-1).cpu().numpy().astype(np.float32)

    def logit_scale(self) -> float:
        self.load()
        return float(self.model.logit_scale.exp().item())

    def zero_shot(self, image_embeddings: np.ndarray, normal_prompts: Optional[List[str]] = None,
                  anomaly_prompts: Optional[List[str]] = None) -> np.ndarray:
        """Prompt-ensemble zero-shot probabilities ``[p_normal, p_anomaly]``."""
        normal = self.encode_texts(normal_prompts or NORMAL_PROMPTS).mean(axis=0)
        anomaly = self.encode_texts(anomaly_prompts or ANOMALY_PROMPTS).mean(axis=0)
        text = np.stack([normal / np.linalg.norm(normal), anomaly / np.linalg.norm(anomaly)])
        logits = self.logit_scale() * image_embeddings @ text.T
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        return (probs / probs.sum(axis=1, keepdims=True)).astype(np.float32)
