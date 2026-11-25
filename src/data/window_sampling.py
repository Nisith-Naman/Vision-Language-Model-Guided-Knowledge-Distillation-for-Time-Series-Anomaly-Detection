"""Deterministic sampling of windows for bounded teacher scoring."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def select_time_distributed_windows(
    total_windows: int,
    sequence_meta: Optional[List[Dict[str, Any]]],
    limit: int,
) -> np.ndarray:
    """Select windows across source groups and time without inspecting labels.

    The budget is allocated in proportion to each source group's number of
    available windows. Within each group, selected indices are spread across
    the full ordered sequence of windows. This avoids choosing teacher examples
    based on their ground-truth class.
    """
    total_windows = int(total_windows)
    limit = int(limit)
    if total_windows <= 0 or limit <= 0:
        return np.empty(0, dtype=np.int64)
    if limit >= total_windows:
        return np.arange(total_windows, dtype=np.int64)

    group_parts: Dict[str, List[np.ndarray]] = {}
    for item in sequence_meta or []:
        indices = np.asarray(item.get("window_indices", []), dtype=np.int64).ravel()
        if not len(indices):
            continue
        group = str(item.get("group_id", len(group_parts)))
        group_parts.setdefault(group, []).append(indices)

    if not group_parts:
        return np.linspace(0, total_windows - 1, limit, dtype=np.int64)

    groups = [np.concatenate(parts) for parts in group_parts.values()]
    counts = np.asarray([len(indices) for indices in groups], dtype=np.int64)
    quotas = np.zeros(len(groups), dtype=np.int64)
    active = np.flatnonzero(counts > 0)

    if limit < len(active):
        chosen_groups = np.linspace(0, len(active) - 1, limit, dtype=np.int64)
        quotas[active[chosen_groups]] = 1
    else:
        quotas[active] = 1
        remaining = limit - len(active)
        while remaining > 0:
            capacity = counts - quotas
            eligible = np.flatnonzero(capacity > 0)
            if not len(eligible):
                break
            raw = remaining * capacity[eligible] / capacity[eligible].sum()
            additions = np.minimum(np.floor(raw).astype(np.int64), capacity[eligible])
            quotas[eligible] += additions
            remaining -= int(additions.sum())
            if remaining <= 0:
                break

            capacity = counts - quotas
            eligible = np.flatnonzero(capacity > 0)
            if not len(eligible):
                break
            raw = remaining * capacity[eligible] / capacity[eligible].sum()
            order = eligible[np.argsort(raw - np.floor(raw), kind="stable")[::-1]]
            for group_index in order:
                if remaining <= 0:
                    break
                quotas[group_index] += 1
                remaining -= 1

    selected: List[int] = []
    for indices, quota in zip(groups, quotas):
        if quota:
            positions = np.linspace(0, len(indices) - 1, int(quota), dtype=np.int64)
            selected.extend(indices[positions].tolist())

    result = np.sort(np.unique(np.asarray(selected, dtype=np.int64)))
    if len(result) != min(total_windows, limit):
        # Defensive fallback for malformed sequence metadata. This fallback is
        # still deterministic and label-independent.
        result = np.linspace(0, total_windows - 1, limit, dtype=np.int64)
    return result
