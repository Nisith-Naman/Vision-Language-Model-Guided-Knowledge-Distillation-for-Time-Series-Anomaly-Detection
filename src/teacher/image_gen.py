"""
Time-series window to image conversion utilities for VLM teacher models.

Converts multidimensional time-series windows into visual artifacts (heatmaps or line plots)
suitable for input into Vision-Language Models like CLIP without leaking ground-truth labels.
"""

import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend to prevent GUI thread conflicts
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns
from tqdm import tqdm


def window_to_image(
    window: np.ndarray,
    feature_names: Optional[List[str]] = None,
    mode: str = "heatmap",
    image_size: Tuple[int, int] = (224, 224),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Image.Image:
    """
    Convert a single time-series window (L, F) into a PIL Image.

    IMPORTANT: Ground-truth anomaly labels must NEVER be included in the rendered image.

    Args:
        window: 2D array of shape (L, F) representing L time steps and F features.
        feature_names: List of feature names of length F. If None, defaults to ['F0', 'F1', ...].
        mode: Visual encoding mode ('heatmap' or 'lineplot').
        image_size: Output image dimensions as (width, height), default (224, 224).
        vmin: Minimum value for consistent colorbar/axis scaling.
        vmax: Maximum value for consistent colorbar/axis scaling.

    Returns:
        PIL.Image.Image: Rendered RGB image of exact dimensions `image_size`.

    Raises:
        ValueError: If mode is not 'heatmap' or 'lineplot', or window has invalid dimensions.
    """
    window = np.asarray(window, dtype=np.float32)

    # Handle shape normalization
    if window.ndim == 1:
        window = window[:, np.newaxis]
    elif window.ndim != 2:
        raise ValueError(f"Expected 2D window of shape (L, F), got shape {window.shape}")

    length, num_features = window.shape

    # Prepare feature names
    if feature_names is None or len(feature_names) != num_features:
        feature_names = [f"F_{i}" for i in range(num_features)]

    # Compute figure size in inches targeting image_size at 100 DPI
    target_w, target_h = image_size
    dpi = 100.0
    figsize = (target_w / dpi, target_h / dpi)

    fig, ax = plt.subplots(figsize=figsize, dpi=int(dpi))

    try:
        if mode == "heatmap":
            # Features on Y-axis (rows), Time on X-axis (columns)
            # window is (L, F) -> transpose to (F, L)
            matrix = window.T

            # Auto-compute scaling if not provided
            effective_vmin = vmin if vmin is not None else float(np.nanmin(matrix))
            effective_vmax = vmax if vmax is not None else float(np.nanmax(matrix))
            if effective_vmin == effective_vmax:
                effective_vmax = effective_vmin + 1.0

            # Render heatmap with viridis colormap
            show_yticklabels = feature_names if num_features <= 30 else False
            sns.heatmap(
                matrix,
                cmap="viridis",
                vmin=effective_vmin,
                vmax=effective_vmax,
                cbar=False,
                xticklabels=False,
                yticklabels=show_yticklabels,
                ax=ax,
            )
            ax.set_title("Time Series Window", fontsize=9, pad=6)
            ax.set_xlabel("Time", fontsize=8)
            ax.set_ylabel("Features", fontsize=8)

            if show_yticklabels and num_features > 10:
                ax.tick_params(axis="y", labelsize=6)

        elif mode == "lineplot":
            time_steps = np.arange(length)

            for feat_idx in range(num_features):
                ax.plot(
                    time_steps,
                    window[:, feat_idx],
                    label=feature_names[feat_idx],
                    linewidth=1.2,
                )

            ax.set_title("Time Series Window", fontsize=9, pad=6)
            ax.set_xlabel("Time", fontsize=8)
            ax.set_ylabel("Value", fontsize=8)
            ax.grid(True, linestyle="--", alpha=0.3)

            if vmin is not None and vmax is not None:
                ax.set_ylim(vmin, vmax)

            # Compact legend
            if num_features <= 12:
                ncol = 2 if num_features > 4 else 1
                ax.legend(
                    loc="upper right",
                    fontsize=6,
                    ncol=ncol,
                    framealpha=0.6,
                )

        else:
            raise ValueError(
                f"Unsupported mode: '{mode}'. Expected 'heatmap' or 'lineplot'."
            )

        fig.tight_layout(pad=0.8)
        fig.canvas.draw()

        # Extract RGB buffer from canvas
        rgba_buffer = np.asarray(fig.canvas.buffer_rgba())
        rgb_buffer = rgba_buffer[:, :, :3]
        pil_image = Image.fromarray(rgb_buffer)

        # Enforce exact target size
        if pil_image.size != image_size:
            pil_image = pil_image.resize(image_size, Image.Resampling.LANCZOS)

        return pil_image

    finally:
        plt.close(fig)


def batch_to_images(
    windows: np.ndarray,
    feature_names: Optional[List[str]] = None,
    mode: str = "heatmap",
    image_size: Tuple[int, int] = (224, 224),
    batch_size: int = 64,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> List[Image.Image]:
    """
    Convert a batch of time-series windows to PIL Images with a progress bar.

    Args:
        windows: 3D array of shape (N, L, F) or sequence of 2D windows.
        feature_names: List of feature names of length F.
        mode: Visual encoding mode ('heatmap' or 'lineplot').
        image_size: Target image dimensions (width, height).
        batch_size: Processing batch size hint.

    Returns:
        List[PIL.Image.Image]: List of rendered PIL Images.
    """
    windows_arr = np.asarray(windows)
    num_windows = len(windows_arr)

    # Pre-calculate global min and max when the caller has not supplied a
    # training-fitted range. Train/validation batches should share one range.
    if num_windows > 0:
        vmin = float(np.nanmin(windows_arr)) if vmin is None else float(vmin)
        vmax = float(np.nanmax(windows_arr)) if vmax is None else float(vmax)

    rendered_images: List[Image.Image] = []

    progress_bar = tqdm(
        total=num_windows,
        desc=f"Generating {mode} images",
        unit="window",
    )

    for i in range(num_windows):
        img = window_to_image(
            window=windows_arr[i],
            feature_names=feature_names,
            mode=mode,
            image_size=image_size,
            vmin=vmin,
            vmax=vmax,
        )
        rendered_images.append(img)
        progress_bar.update(1)

    progress_bar.close()
    return rendered_images


def save_image_cache(
    images: List[Image.Image],
    cache_path: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Serialize and save a list of PIL Images and metadata to disk using pickle.

    Args:
        images: List of PIL Image objects.
        cache_path: Target file path on disk.
        metadata: Optional dictionary of metadata (e.g. dataset name, prompts, mode).
    """
    cache_dir = os.path.dirname(os.path.abspath(cache_path))
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    payload = {
        "images": images,
        "metadata": metadata if metadata is not None else {},
    }

    with open(cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_image_cache(cache_path: str) -> Tuple[List[Image.Image], Dict[str, Any]]:
    """
    Load a list of PIL Images and metadata from a disk pickle cache.

    Args:
        cache_path: Path to cached file on disk.

    Returns:
        Tuple[List[PIL.Image.Image], Dict[str, Any]]: Loaded images and metadata dict.

    Raises:
        FileNotFoundError: If cache_path does not exist.
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache file not found at: {cache_path}")

    with open(cache_path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict) and "images" in data:
        return data["images"], data.get("metadata", {})
    elif isinstance(data, tuple) and len(data) == 2:
        return data[0], data[1]
    elif isinstance(data, list):
        return data, {}
    else:
        return [data], {}
