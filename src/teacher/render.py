"""Fast, label-free rendering of multivariate windows into 224x224 RGB images.

The earlier Matplotlib heatmap used one global colour range taken from the
training data. With standardized sensors the range is dominated by a few
extreme values (e.g. -95..108 on Daphnet), so most windows rendered as an almost
uniform image and CLIP could not separate them. These renderers clip each
standardized channel to ``[-clip, clip]`` and draw directly with PIL/NumPy,
which is also ~20x faster than creating a Matplotlib figure per window.

Styles
------
``lines``   one horizontal band per channel, each drawn as a coloured polyline
            (the "chart" form CLIP has seen in web images).
``heatmap`` channels x time, viridis colour map, nearest-neighbour upscaling
            (keeps every channel visible when there are many of them).
``auto``    ``lines`` for <= 16 channels, otherwise ``heatmap``.

Ground-truth labels are never drawn.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
from PIL import Image, ImageDraw

IMAGE_SIZE = 224
_PALETTE = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40), (148, 103, 189),
    (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34), (23, 190, 207),
]


def _viridis_lut() -> np.ndarray:
    try:
        import matplotlib

        cmap = matplotlib.colormaps["viridis"]
        return (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    except Exception:  # pragma: no cover - matplotlib is a project dependency
        ramp = np.linspace(0, 255, 256).astype(np.uint8)
        return np.stack([ramp // 3, ramp, 255 - ramp], axis=1)


_VIRIDIS = _viridis_lut()


def resolve_style(style: str, n_features: int) -> str:
    style = (style or "auto").lower()
    if style == "auto":
        return "lines" if n_features <= 16 else "heatmap"
    if style not in ("lines", "heatmap"):
        raise ValueError(f"Unknown render style {style!r}")
    return style


def render_window(window: np.ndarray, style: str = "auto", clip: float = 4.0,
                  size: int = IMAGE_SIZE) -> np.ndarray:
    """Render one standardized window ``(L, F)`` to a ``(size, size, 3)`` uint8 array."""
    w = np.nan_to_num(np.asarray(window, dtype=np.float32), nan=0.0, posinf=clip, neginf=-clip)
    if w.ndim == 1:
        w = w[:, None]
    length, n_feat = w.shape
    w = np.clip(w, -clip, clip)
    style = resolve_style(style, n_feat)

    if style == "heatmap":
        idx = ((w.T + clip) / (2 * clip) * 255).round().astype(np.uint8)  # (F, L)
        rgb = _VIRIDIS[idx]
        return np.asarray(Image.fromarray(rgb).resize((size, size), Image.Resampling.NEAREST))

    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    margin = 4
    band = (size - 2 * margin) / n_feat
    xs = np.linspace(margin, size - margin - 1, length)
    for f in range(n_feat):
        top = margin + f * band
        mid = top + band / 2
        draw.line([(margin, mid), (size - margin, mid)], fill=(225, 225, 225), width=1)
        ys = mid - (w[:, f] / clip) * (band / 2 - 1)
        draw.line(list(zip(xs.tolist(), ys.tolist())), fill=_PALETTE[f % len(_PALETTE)], width=1)
    return np.asarray(img)


def render_batch(windows: Iterable[np.ndarray], style: str = "auto", clip: float = 4.0) -> np.ndarray:
    """Render many windows to a ``(N, 224, 224, 3)`` uint8 array."""
    return np.stack([render_window(w, style=style, clip=clip) for w in windows], axis=0)


def to_pil(images: np.ndarray) -> List[Image.Image]:
    return [Image.fromarray(np.asarray(im, dtype=np.uint8)) for im in images]


def render_spectrogram(window: np.ndarray, clip: float = 4.0, size: int = IMAGE_SIZE) -> np.ndarray:
    """Short-time Fourier magnitude of each standardized channel, stacked vertically.

    Rows: channel x frequency bin (low frequencies at the top of each band);
    columns: time frames. Freezing of gait, bearing faults and slugging are
    rhythmic events that are clearer in frequency than in amplitude.
    Magnitudes use a fixed scale (no per-window normalisation), so absolute
    energy differences between windows are preserved. Labels are never drawn.
    """
    w = np.clip(np.nan_to_num(np.asarray(window, dtype=np.float32)), -clip, clip)
    if w.ndim == 1:
        w = w[:, None]
    length, n_feat = w.shape
    nperseg = int(max(8, min(64, 2 ** int(np.log2(max(8, length // 4))))))
    hop = max(1, nperseg // 4)
    starts = np.arange(0, max(1, length - nperseg + 1), hop)
    frames = np.stack([w[s:s + nperseg] for s in starts], axis=0)            # (T', nperseg, F)
    if frames.shape[1] < nperseg:
        frames = np.pad(frames, ((0, 0), (0, nperseg - frames.shape[1]), (0, 0)))
    spec = np.abs(np.fft.rfft(frames * np.hanning(nperseg)[None, :, None], axis=1))  # (T', bins, F)
    mag = np.log1p(spec) / np.log1p(clip * nperseg / 2)
    img = np.clip(mag, 0, 1).transpose(2, 1, 0).reshape(n_feat * spec.shape[1], len(starts))
    rgb = _VIRIDIS[(img * 255).round().astype(np.uint8)]
    return np.asarray(Image.fromarray(rgb).resize((size, size), Image.Resampling.NEAREST))


def render_views(windows: Iterable[np.ndarray], views: Iterable[str], style: str = "heatmap",
                 clip: float = 4.0) -> dict:
    """Render a batch in several views: 'heatmap'/'lines'/'auto' or 'spectrogram'."""
    windows = list(windows)
    out = {}
    for v in views:
        if v == "spectrogram":
            out[v] = np.stack([render_spectrogram(w, clip) for w in windows])
        else:
            out[v] = render_batch(windows, style if v == "primary" else v, clip)
    return out
