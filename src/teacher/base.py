"""
Abstract base class definition for Vision-Language Model (VLM) teacher models.

Teacher models evaluate visual representations of time-series windows against
paired textual prompts to provide soft anomaly probabilities for student distillation.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from PIL import Image


class BaseTeacher(ABC):
    """
    Abstract base class for all teacher models.

    Defines the contract for loading pretrained models, scoring visual representations
    of time-series windows against semantic textual prompts, and ensuring model weights
    remain frozen during distillation.
    """

    @abstractmethod
    def score_window(
        self,
        image: Image.Image,
        normal_prompt: str,
        anomaly_prompt: str,
    ) -> np.ndarray:
        """
        Score a single time-series window image against normal and anomaly prompts.

        Args:
            image: Rendered PIL Image of the time-series window.
            normal_prompt: Text description representing nominal / normal behavior.
            anomaly_prompt: Text description representing anomalous behavior.

        Returns:
            np.ndarray: Probability vector p_T = [p_normal, p_anomaly] of shape (2,)
                        where p_normal + p_anomaly == 1.0.
        """
        pass

    @abstractmethod
    def load_model(self) -> None:
        """
        Load pretrained model weights and freeze all parameters.

        Implementations must set the model to evaluation mode (eval()) and
        ensure requires_grad is False for all parameters.
        """
        pass

    def is_frozen(self) -> bool:
        """
        Check whether all underlying model parameters are frozen (requires_grad is False).

        Returns:
            bool: True if model is loaded and all parameters have requires_grad=False;
                  False if model is not yet loaded or has unfrozen parameters.
        """
        model = getattr(self, "model", None)
        if model is None:
            return False

        if hasattr(model, "parameters"):
            params = list(model.parameters())
            if not params:
                return False
            return not any(p.requires_grad for p in params)

        return True
