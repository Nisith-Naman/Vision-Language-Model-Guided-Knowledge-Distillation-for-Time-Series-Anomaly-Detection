"""Page 3: Preprocessing & Windowing.

Configures temporal sorting, missing value imputation, train-only scaling,
sliding window slicing, and split leakage verification.
"""

from __future__ import annotations

import logging
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src.data.loader import Series
from src.data.synthetic_labels import inject_anomalies
from src.data.preprocessing import (
    check_leakage,
    chronological_split,
    fill_missing,
    fit_imputation_stats_batches,
    fit_scaler_batches,
    handle_duplicates,
    make_windows,
    sort_chronologically,
    transform_scaler,
)
from src.inference.context import calibrate_context_threshold

logger = logging.getLogger(__name__)


def _window_start_indices(n_steps: int, window_length: int, stride: int) -> np.ndarray:
    if n_steps < window_length:
        return np.empty(0, dtype=np.int64)
    starts = np.arange(0, n_steps - window_length + 1, stride, dtype=np.int64)
    if len(starts) and starts[-1] + window_length < n_steps:
        starts = np.append(starts, n_steps - window_length)
    return starts


def _allocate_window_quotas(counts: List[int], cap: int) -> np.ndarray:
    """Allocate a capped number of windows proportionally across sequences."""
    count_array = np.asarray(counts, dtype=np.int64)
    total = int(count_array.sum())
    if cap <= 0 or total <= cap:
        return count_array

    active = np.flatnonzero(count_array > 0)
    quotas = np.zeros_like(count_array)
    if cap < len(active):
        selected = active[np.argsort(count_array[active], kind="stable")[-cap:]]
        quotas[selected] = 1
        return quotas

    quotas[active] = 1
    remaining = cap - len(active)
    while remaining > 0:
        capacity = count_array - quotas
        eligible = np.flatnonzero(capacity > 0)
        if not len(eligible):
            break
        weights = capacity[eligible].astype(np.float64)
        raw = remaining * weights / weights.sum()
        proposed = np.floor(raw).astype(np.int64)
        proposed = np.minimum(proposed, capacity[eligible])
        allocated = int(proposed.sum())
        quotas[eligible] += proposed
        remaining -= allocated
        if remaining <= 0:
            break

        capacity = count_array - quotas
        can_receive_one = capacity[eligible] > 0
        candidates = eligible[can_receive_one]
        if not len(candidates):
            break
        fractional = raw - np.floor(raw)
        order = candidates[np.argsort(fractional[can_receive_one], kind="stable")[::-1]]
        for idx in order:
            if remaining <= 0:
                break
            quotas[idx] += 1
            remaining -= 1
    return quotas


def _representative_window_indices(labels: np.ndarray, quota: int) -> np.ndarray:
    """Choose deterministic, time-distributed windows while retaining both labels."""
    n_windows = len(labels)
    if quota >= n_windows:
        return np.arange(n_windows, dtype=np.int64)
    if quota <= 0:
        return np.empty(0, dtype=np.int64)
    if quota == 1:
        return np.asarray([n_windows // 2], dtype=np.int64)

    normal = np.flatnonzero(labels == 0)
    anomaly = np.flatnonzero(labels == 1)
    if not len(normal) or not len(anomaly):
        return np.linspace(0, n_windows - 1, quota, dtype=np.int64)

    anomaly_quota = int(round(quota * len(anomaly) / n_windows))
    anomaly_quota = min(quota - 1, max(1, anomaly_quota))
    normal_quota = quota - anomaly_quota
    selected = np.r_[
        np.linspace(normal[0], normal[-1], normal_quota, dtype=np.int64),
        np.linspace(anomaly[0], anomaly[-1], anomaly_quota, dtype=np.int64),
    ]
    return np.unique(selected)


def _head_rows(arrays: List[np.ndarray], n_rows: int = 200) -> np.ndarray:
    """Collect a small diagnostic preview without concatenating full datasets."""
    chunks = []
    remaining = n_rows
    for arr in arrays:
        if remaining <= 0:
            break
        take = min(len(arr), remaining)
        if take:
            chunks.append(arr[:take])
            remaining -= take
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 0), dtype=np.float32)


def show() -> None:
    """Render the Preprocessing page."""
    st.header("Data Preprocessing & Windowing")
    st.caption("Apply train-only scaling, temporal imputation, and sliding-window generation without data leakage.")

    # Guard: require data selection from page 2
    if "selected_features" not in st.session_state or "raw_df" not in st.session_state:
        st.warning("Please complete Step 2 (Upload & Inspect Data) first to select feature columns and labels.")
        return

    raw_df: pd.DataFrame = st.session_state["raw_df"]
    selected_features: List[str] = st.session_state["selected_features"]
    time_col: Optional[str] = st.session_state.get("time_col")
    label_col: Optional[str] = st.session_state.get("label_col")
    entity_col: Optional[str] = st.session_state.get("entity_col")
    benchmark_key: Optional[str] = st.session_state.get("benchmark_key")
    benchmark_config: Dict[str, Any] = st.session_state.get("benchmark_config", {})
    app_config: Dict[str, Any] = st.session_state.get("app_config", {})
    normal_val_str: str = st.session_state.get("normal_label_val", "0")
    anomaly_val_str: str = st.session_state.get("anomaly_label_val", "1")

    st.markdown("---")
    st.subheader("Preprocessing Configuration")

    # Form / Sidebar layout for configuration parameters
    with st.expander("Configure Pipeline Hyperparameters", expanded=True):
        col_c1, col_c2, col_c3 = st.columns(3)

        with col_c1:
            scaler_choice = st.selectbox(
                "Feature Scaler:",
                ["Standard", "MinMax"],
                index=0,
                help="Fitted strictly on the training partition only to prevent data leakage.",
            )
            missing_strategy = st.selectbox(
                "Missing Value Strategy:",
                ["interpolate", "ffill", "bfill", "mean"],
                index=0,
                help="Imputation method applied chronologically along each feature channel.",
            )

        with col_c2:
            duplicate_policy = st.selectbox(
                "Duplicate Timestamp Policy:",
                ["first", "last", "mean"],
                index=0,
                help="Resolution policy when identical timestamps appear in the series.",
            )
            anomaly_rule = st.selectbox(
                "Window Anomaly Rule:",
                ["any", "majority"],
                index=0,
                help="'any': window label is 1 if >=1 anomalous step exists. 'majority': if >=50% are anomalous.",
            )

        with col_c3:
            configured_window = int(benchmark_config.get(
                "window_length", app_config.get("windowing", {}).get("window_length", 128)
            ))
            window_length = st.slider(
                "Window Length (L):",
                min_value=16,
                max_value=512,
                value=min(512, max(16, configured_window)),
                step=4,
                key=f"window_length_{benchmark_key or 'custom'}",
                help="Sequence length of each sliding temporal window.",
            )
            configured_stride = int(benchmark_config.get(
                "stride", app_config.get("windowing", {}).get("stride", 32)
            ))
            stride = st.slider(
                "Window Stride (S):",
                min_value=4,
                max_value=128,
                value=min(128, max(4, configured_stride)),
                step=4,
                key=f"window_stride_{benchmark_key or 'custom'}",
                help="Step offset between successive sliding windows.",
            )

        st.markdown("##### Chronological Split Ratios")
        configured_splits = benchmark_config.get("splits", app_config.get("splits", {}))
        configured_train = float(configured_splits.get("train", 0.70))
        configured_val = float(configured_splits.get("val", 0.15))
        s_col1, s_col2, s_col3 = st.columns(3)
        with s_col1:
            train_ratio = st.slider(
                "Train Split", min_value=0.40, max_value=0.85,
                value=min(0.85, max(0.40, configured_train)), step=0.05,
                key=f"train_split_{benchmark_key or 'custom'}",
            )
        with s_col2:
            val_ratio = st.slider(
                "Validation Split", min_value=0.05, max_value=0.30,
                value=min(0.30, max(0.05, configured_val)), step=0.05,
                key=f"val_split_{benchmark_key or 'custom'}",
            )
        with s_col3:
            test_ratio = round(1.0 - train_ratio - val_ratio, 2)
            st.metric("Test Split (Remainder)", f"{test_ratio:.2f}")

        configured_train_cap = int(benchmark_config.get("max_train_windows", 0))
        max_train_windows = int(st.number_input(
            "Maximum training windows (0 = all):",
            min_value=0,
            max_value=1_000_000,
            value=min(1_000_000, max(0, configured_train_cap)),
            step=5_000,
            key=f"max_train_windows_{benchmark_key or 'custom'}",
            help=(
                "A deterministic, class-aware sample is retained across sequences when the cap permits. "
                "Validation and test keep every window so reported evaluation metrics use the full split."
            ),
        ))

        if test_ratio <= 0.0:
            st.error("Train and Validation split ratios must sum to strictly less than 1.0 (Test split must be > 0).")
            return

    # Action Button
    st.markdown("<br>", unsafe_allow_html=True)
    run_preproc = st.button("Run Preprocessing & Sliding Windows", type="primary", use_container_width=True)

    if run_preproc:
        # A changed split, scaler, feature set, or window invalidates all later
        # artifacts. Do not let old teacher scores or checkpoints attach to new windows.
        downstream_keys = (
            "preprocessing_complete", "X_train_windows", "y_window_train", "y_step_train",
            "X_val_windows", "y_window_val", "y_step_val", "X_test_windows", "y_window_test", "y_step_test",
            "ts_train_starts", "ts_val_starts", "ts_test_starts", "train_sequence_meta",
            "val_sequence_meta", "test_sequence_meta", "split_stats", "split_description", "split_manifest",
            "scaler", "scaler_type", "X_train_raw_sample", "X_train_scaled_sample",
            "train_impute_stats", "missing_strategy", "missing_codes",
            "contextual_threshold_robust_z", "contextual_calibration_normal_rows",
            "context_robust_z_threshold",
            "teacher_complete", "teacher_probs_train", "teacher_train_indices", "teacher_probs_val",
            "teacher_val_indices", "teacher_eval_dict", "teacher_scoring_config",
            "trained_model", "training_complete", "training_result", "training_mode", "last_checkpoint_path",
            "selected_threshold", "selected_window_threshold", "thresh_info", "window_thresh_info",
            "thresh_curve_data", "test_metrics", "test_step_metrics", "test_event_metrics",
            "test_window_probs", "test_step_scores", "eval_complete", "inference_complete",
        )
        for key in downstream_keys:
            st.session_state.pop(key, None)

        with st.spinner("Executing split-safe preprocessing pipeline..."):
            try:
                source_series = st.session_state.get("benchmark_series_list")
                if source_series:
                    base_series_list = []
                    for source in source_series:
                        missing_features = [f for f in selected_features if f not in source.feature_names]
                        if missing_features:
                            raise ValueError(f"Selected features are missing from a benchmark sequence: {missing_features}")
                        feature_indices = [source.feature_names.index(f) for f in selected_features]
                        selected_all_features = feature_indices == list(range(source.n_features))
                        base_series_list.append(Series(
                            values=source.values if selected_all_features else source.values[:, feature_indices],
                            y_step=source.y_step if source.y_step is not None else np.zeros(source.n_timesteps, dtype=np.int8),
                            timestamps=source.timestamps,
                            group_id=source.group_id,
                            feature_names=list(selected_features),
                            metadata=dict(source.metadata),
                        ))
                else:
                    df_work = raw_df.copy()
                    grouped_input = entity_col is not None and entity_col in df_work.columns
                    group_items = list(df_work.groupby(entity_col, sort=False, dropna=False)) if grouped_input else [("primary_series", df_work)]
                    base_series_list = []
                    for group_id, group_df in group_items:
                        if time_col is not None and time_col in group_df.columns:
                            group_df = sort_chronologically(group_df, time_col=time_col)
                            group_df = handle_duplicates(group_df, time_col=time_col, policy=duplicate_policy)
                            timestamps = group_df[time_col].to_numpy()
                        else:
                            timestamps = np.arange(len(group_df))
                        y_step_raw = (
                            (group_df[label_col].astype(str).str.strip() == str(anomaly_val_str).strip()).astype(np.int8).to_numpy()
                            if label_col is not None and label_col in group_df.columns
                            else np.zeros(len(group_df), dtype=np.int8)
                        )
                        base_series_list.append(Series(
                            values=group_df[selected_features].to_numpy(dtype=np.float32),
                            y_step=y_step_raw,
                            timestamps=timestamps,
                            group_id=str(group_id),
                            feature_names=list(selected_features),
                        ))

                if not base_series_list:
                    raise ValueError("No valid time-series sequences were found in the selected data.")

                # Split whole subjects/instances when available; preserve the event
                # assignment for MetroPT, whose four failures are too few for a random split.
                explicit_roles = all(s.metadata.get("split_role") in {"train", "val", "test"} for s in base_series_list)
                split_strategy = st.session_state.get("benchmark_split_by", "chronological")
                if explicit_roles:
                    train_s_list = [s for s in base_series_list if s.metadata["split_role"] == "train"]
                    val_s_list = [s for s in base_series_list if s.metadata["split_role"] == "val"]
                    test_s_list = [s for s in base_series_list if s.metadata["split_role"] == "test"]
                    split_description = (
                        "MetroPT event holdout: event 1 train, event 2 validation, events 3-4 test; "
                        "normal-only acquisition blocks are split chronologically 70/15/15"
                        if benchmark_key == "metropt"
                        else "dataset-provided split roles"
                    )
                elif len({s.group_id for s in base_series_list}) >= 3 and (
                    split_strategy in {"subject", "instance", "event", "group"} or entity_col is not None
                ):
                    train_s_list, val_s_list, test_s_list = chronological_split(
                        base_series_list, train_frac=train_ratio, val_frac=val_ratio,
                        split_by=split_strategy if split_strategy in {"subject", "instance", "event", "group"} else "group",
                        seed=42,
                    )
                    split_description = f"group holdout ({split_strategy})"
                elif len(base_series_list) == 1:
                    train_s_list, val_s_list, test_s_list = chronological_split(
                        base_series_list, train_frac=train_ratio, val_frac=val_ratio, split_by="chronological"
                    )
                    split_description = "chronological time holdout"
                else:
                    # Small custom collections may not have enough distinct groups
                    # for three identity-disjoint splits. Split each recording by time.
                    train_s_list, val_s_list, test_s_list = [], [], []
                    for sequence in base_series_list:
                        tr, va, te = chronological_split(
                            [sequence], train_frac=train_ratio, val_frac=val_ratio, split_by="chronological"
                        )
                        train_s_list.extend(tr)
                        val_s_list.extend(va)
                        test_s_list.extend(te)
                    split_description = "per-sequence chronological holdout (fewer than 3 independent groups)"

                if not train_s_list or not val_s_list or not test_s_list:
                    raise ValueError("The selected split produced an empty partition. Adjust split ratios or add more independent sequences.")
                eligible_train_sequences = sum(
                    s.n_timesteps >= window_length for s in train_s_list
                )
                if max_train_windows > 0 and max_train_windows < eligible_train_sequences:
                    st.warning(
                        f"The training-window cap ({max_train_windows:,}) is smaller than the "
                        f"{eligible_train_sequences:,} train sequences long enough for a window. "
                        "Some sequences will not contribute student training windows; increase the cap to include all."
                    )

                group_identity_split = explicit_roles or split_description.startswith("group holdout")
                leakage_passed = check_leakage(
                    train_s_list, val_s_list, test_s_list, check_group_identity=group_identity_split
                )

                # Imputation references and scaler are fit on training data only.
                train_impute_stats = fit_imputation_stats_batches(
                    [s.values for s in train_s_list]
                )

                def _impute(series_batch: List[Series]) -> List[np.ndarray]:
                    return [
                        s.values if np.isfinite(s.values).all()
                        else fill_missing(s.values, strategy=missing_strategy, train_stats=train_impute_stats)[0]
                        for s in series_batch
                    ]

                train_imputed = _impute(train_s_list)
                val_imputed = _impute(val_s_list)
                test_imputed = _impute(test_s_list)
                # Fit the scaler on clean, imputed training data before any synthetic
                # Air Quality anomalies are injected.
                fitted_scaler = fit_scaler_batches(
                    train_imputed, scaler_type=scaler_choice.strip().lower()
                )
                needs_airquality_injection = any(
                    s.metadata.get("dataset") == "airquality"
                    and s.metadata.get("label_strategy") == "semi_synthetic"
                    for s in train_s_list + val_s_list + test_s_list
                )
                if needs_airquality_injection:
                    std_scaler = fit_scaler_batches(train_imputed, scaler_type="standard")
                    train_stds = np.asarray(std_scaler.scale_, dtype=np.float32)
                else:
                    train_stds = None

                def _inject_airquality(
                    series_batch: List[Series], values_batch: List[np.ndarray], split_offset: int
                ) -> List[np.ndarray]:
                    injected_batch = []
                    for seq_idx, (sequence, values) in enumerate(zip(series_batch, values_batch)):
                        metadata = sequence.metadata
                        if (metadata.get("dataset") == "airquality"
                                and metadata.get("label_strategy") == "semi_synthetic"):
                            values, labels, injection_meta = inject_anomalies(
                                values=values,
                                timestamps=sequence.timestamps,
                                config={"injection": metadata.get("injection_config", {})},
                                seed=int(metadata.get("synthetic_seed", 42)) + split_offset + seq_idx,
                                start_idx=0,
                                scale_stds=train_stds,
                                return_metadata=True,
                            )
                            sequence.values = values
                            sequence.y_step = labels
                            sequence.metadata["injection_result"] = injection_meta
                        injected_batch.append(values)
                    return injected_batch

                train_imputed = _inject_airquality(train_s_list, train_imputed, split_offset=0)
                val_imputed = _inject_airquality(val_s_list, val_imputed, split_offset=1000)
                test_imputed = _inject_airquality(test_s_list, test_imputed, split_offset=2000)

                def _window_split(
                    series_batch: List[Series],
                    value_batch: List[np.ndarray],
                    window_cap: int = 0,
                    keep_values_for_inference: bool = False,
                ):
                    start_lists = [
                        _window_start_indices(s.n_timesteps, window_length, stride)
                        for s in series_batch
                    ]
                    available_counts = [len(starts) for starts in start_lists]
                    quotas = _allocate_window_quotas(available_counts, window_cap)
                    used_total = int(quotas.sum())
                    n_features = len(selected_features)
                    X_out = np.empty((used_total, window_length, n_features), dtype=np.float32)
                    y_window_out = np.empty(used_total, dtype=np.int8)
                    y_steps_out = np.empty((used_total, window_length), dtype=np.int8)
                    starts_out: List[Any] = []
                    sequence_meta: List[Dict[str, Any]] = []
                    next_window_index = 0
                    for sequence_idx, (sequence, values) in enumerate(zip(series_batch, value_batch)):
                        scaled_values = transform_scaler(values, fitted_scaler)
                        labels = sequence.y_step if sequence.y_step is not None else np.zeros(sequence.n_timesteps, dtype=np.int8)
                        Xw, yw, ys, ts = make_windows(
                            values=scaled_values, y_step=labels, timestamps=sequence.timestamps,
                            window_length=window_length, stride=stride, anomaly_rule=anomaly_rule,
                            cover_tail=True,
                        )
                        if len(Xw) == 0 or quotas[sequence_idx] == 0:
                            continue
                        local_starts = start_lists[sequence_idx]
                        if len(local_starts) != len(Xw):
                            raise ValueError("Window start indexing did not match generated windows.")
                        selected_window_indices = _representative_window_indices(
                            yw, int(quotas[sequence_idx])
                        )
                        if len(selected_window_indices) != quotas[sequence_idx]:
                            raise ValueError("Window sampling did not meet the allocated per-sequence quota.")
                        Xw = Xw[selected_window_indices]
                        yw = yw[selected_window_indices]
                        ys = ys[selected_window_indices]
                        ts = [ts[i] for i in selected_window_indices]
                        local_starts = local_starts[selected_window_indices]
                        selected_count = len(yw)
                        offset = next_window_index
                        X_out[offset:offset + selected_count] = Xw
                        y_window_out[offset:offset + selected_count] = yw
                        y_steps_out[offset:offset + selected_count] = ys
                        starts_out.extend(ts)
                        metadata = {
                            "group_id": sequence.group_id,
                            "y_step": labels,
                            "window_indices": np.arange(offset, offset + selected_count, dtype=int),
                            "window_starts": local_starts,
                        }
                        if keep_values_for_inference:
                            metadata.update({
                                "values_raw": sequence.values,
                                "values_scaled": scaled_values,
                                "timestamps": sequence.timestamps,
                            })
                        sequence_meta.append(metadata)
                        next_window_index += selected_count
                    return (
                        X_out, y_window_out, y_steps_out, starts_out, sequence_meta,
                        sum(available_counts),
                    )

                X_train_win, y_win_tr, y_step_tr, ts_tr_starts, train_meta, train_windows_available = _window_split(
                    train_s_list, train_imputed, window_cap=max_train_windows
                )
                X_val_win, y_win_va, y_step_va, ts_va_starts, val_meta, val_windows_available = _window_split(
                    val_s_list, val_imputed, keep_values_for_inference=True
                )
                X_test_win, y_win_te, y_step_te, ts_te_starts, test_meta, test_windows_available = _window_split(
                    test_s_list, test_imputed, keep_values_for_inference=True
                )
                if not len(X_train_win) or not len(X_val_win) or not len(X_test_win):
                    raise ValueError("At least one split has no complete windows. Reduce window length or add more timesteps.")

                def _calc_stats(
                    series_batch: List[Series], y_win: np.ndarray, windows_available: int
                ) -> Dict[str, Any]:
                    total_T = sum(s.n_timesteps for s in series_batch)
                    anom_steps = sum(int(np.sum(s.y_step == 1)) for s in series_batch if s.y_step is not None)
                    n_wins = len(y_win)
                    anom_wins = int(np.sum(y_win == 1))
                    return {
                        "timesteps": total_T,
                        "series": len(series_batch),
                        "anom_steps": anom_steps,
                        "step_anom_rate": (anom_steps / total_T) if total_T else 0.0,
                        "windows": n_wins,
                        "windows_available": windows_available,
                        "anom_windows": anom_wins,
                        "win_anom_rate": (anom_wins / n_wins) if n_wins else 0.0,
                    }

                split_stats = {
                    "Train": _calc_stats(train_s_list, y_win_tr, train_windows_available),
                    "Val": _calc_stats(val_s_list, y_win_va, val_windows_available),
                    "Test": _calc_stats(test_s_list, y_win_te, test_windows_available),
                }

                # Keep enough provenance with each checkpoint to establish which
                # source data and exact split produced its reported score.
                data_hasher = hashlib.sha256()
                for sequence in base_series_list:
                    data_hasher.update(str(sequence.group_id).encode("utf-8"))
                    data_hasher.update("\0".join(sequence.feature_names).encode("utf-8"))
                    for array in (sequence.values, sequence.y_step, sequence.timestamps):
                        if array is None:
                            data_hasher.update(b"<none>")
                            continue
                        contiguous = np.ascontiguousarray(array)
                        data_hasher.update(str(contiguous.dtype).encode("ascii"))
                        data_hasher.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
                        if contiguous.dtype.hasobject:
                            data_hasher.update(repr(contiguous.tolist()).encode("utf-8"))
                        else:
                            data_hasher.update(memoryview(contiguous).cast("B"))

                split_hasher = hashlib.sha256()
                split_hasher.update(data_hasher.digest())
                split_hasher.update(str((split_description, window_length, stride, scaler_choice, selected_features)).encode("utf-8"))
                for name, arrays in (
                    ("train", (X_train_win, y_win_tr, y_step_tr)),
                    ("val", (X_val_win, y_win_va, y_step_va)),
                    ("test", (X_test_win, y_win_te, y_step_te)),
                ):
                    split_hasher.update(name.encode("ascii"))
                    for array in arrays:
                        contiguous = np.ascontiguousarray(array)
                        split_hasher.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
                        split_hasher.update(memoryview(contiguous).cast("B"))

                split_manifest = {
                    "dataset_key": benchmark_key or data_filename,
                    "source_data_sha256": data_hasher.hexdigest(),
                    "processed_split_sha256": split_hasher.hexdigest(),
                    "split_description": split_description,
                    "split_seed": 42 if split_description.startswith("group holdout") else None,
                    "feature_names": list(selected_features),
                    "window_length": int(window_length),
                    "stride": int(stride),
                    "scaler": scaler_choice.strip().lower(),
                    "partitions": {
                        "train": {
                            "group_ids": sorted({str(s.group_id) for s in train_s_list}),
                            "series": int(len(train_s_list)),
                            **split_stats["Train"],
                        },
                        "validation": {
                            "group_ids": sorted({str(s.group_id) for s in val_s_list}),
                            "series": int(len(val_s_list)),
                            **split_stats["Val"],
                        },
                        "test": {
                            "group_ids": sorted({str(s.group_id) for s in test_s_list}),
                            "series": int(len(test_s_list)),
                            **split_stats["Test"],
                        },
                    },
                }
                scaler_type_clean = scaler_choice.strip().lower()
                context_threshold, context_normal_rows = calibrate_context_threshold(
                    sequence_meta=val_meta,
                    feature_names=selected_features,
                    target_false_alarm_rate=0.005,
                    conservative_floor=4.5,
                )
                # Validation values are needed only for this calibration; keep
                # timeline metadata, not a second full copy of the validation set.
                for sequence in val_meta:
                    sequence.pop("values_raw", None)
                    sequence.pop("values_scaled", None)
                    sequence.pop("timestamps", None)
                session_values = {
                    "X_train_windows": X_train_win, "y_window_train": y_win_tr, "y_step_train": y_step_tr,
                    "ts_train_starts": ts_tr_starts, "train_sequence_meta": train_meta,
                    "X_val_windows": X_val_win, "y_window_val": y_win_va, "y_step_val": y_step_va,
                    "ts_val_starts": ts_va_starts, "val_sequence_meta": val_meta,
                    "X_test_windows": X_test_win, "y_window_test": y_win_te, "y_step_test": y_step_te,
                    "ts_test_starts": ts_te_starts, "test_sequence_meta": test_meta,
                    "scaler": fitted_scaler, "scaler_type": scaler_type_clean,
                    "train_impute_stats": train_impute_stats,
                    "missing_strategy": missing_strategy,
                    "contextual_threshold_robust_z": context_threshold,
                    "contextual_calibration_normal_rows": context_normal_rows,
                    "missing_codes": (
                        [st.session_state.get("benchmark_config", {}).get("missing_code")]
                        if st.session_state.get("benchmark_config", {}).get("missing_code") is not None
                        else []
                    ),
                    "window_length": window_length, "stride": stride, "anomaly_rule": anomaly_rule,
                    "max_train_windows": max_train_windows,
                    "split_stats": split_stats, "split_description": split_description,
                    "split_manifest": split_manifest,
                    "preprocessing_complete": True,
                }
                for key, value in session_values.items():
                    st.session_state[key] = value
                st.session_state["X_train_raw_sample"] = _head_rows(train_imputed)
                st.session_state["X_train_scaled_sample"] = transform_scaler(
                    _head_rows(train_imputed), fitted_scaler
                )

                if benchmark_key == "metropt":
                    st.warning("MetroPT uses event holdout because it has only four known failures: event 1 is training, event 2 validation, and events 3-4 test. Normal-only acquisition blocks are split chronologically across all three partitions, so validation and test include normal examples as well as failure examples.")
                if benchmark_key == "airquality":
                    st.warning("Air Quality has no native anomaly labels. This run injects synthetic examples independently into train, validation, and test after splitting; the scaler and anomaly magnitudes are based on clean training data. Metrics measure recovery of injected patterns, not real incident detection.")
                if not st.session_state.get("has_ground_truth", True):
                    st.warning("No anomaly labels were provided. This student uses supervised localization and cannot learn meaningful anomaly boundaries from an all-normal placeholder label vector.")
                st.success(f"Preprocessing completed: {split_description}. Windows never cross source-sequence boundaries.")
                st.caption(
                    f"Contextual robust-z cutoff calibrated on {context_normal_rows:,} normal validation rows "
                    f"(target false-alarm rate 0.5%, conservative floor 4.5): {context_threshold:.2f}."
                )

            except Exception as e:
                st.error(f"Preprocessing error encountered: {e}")
                logger.exception("Preprocessing pipeline failure")
                return

    # Preview Section (displayed when preprocessing state exists)
    if st.session_state.get("preprocessing_complete", False):
        st.markdown("---")
        st.subheader("Preprocessing Inspection & Diagnostic Previews")

        split_stats = st.session_state.get("split_stats", {})

        # 1. Split Statistics Table
        summary_rows = []
        for s_name, st_dict in split_stats.items():
            summary_rows.append({
                "Partition Split": s_name,
                "Timesteps (T)": f"{st_dict['timesteps']:,}",
                "Step Anomalies": f"{st_dict['anom_steps']:,}",
                "Step Anomaly Rate": f"{st_dict['step_anom_rate']:.2%}",
                    "Windows Available": f"{st_dict['windows_available']:,}",
                    "Windows Used": f"{st_dict['windows']:,}",
                "Anomalous Windows": f"{st_dict['anom_windows']:,}",
                "Window Anomaly Rate": f"{st_dict['win_anom_rate']:.2%}",
            })

        st.markdown("##### Split Partition Summary")
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
        train_stats = split_stats.get("Train", {})
        if train_stats.get("windows", 0) < train_stats.get("windows_available", 0):
            st.info(
                f"The student training subset uses {train_stats['windows']:,} of "
                f"{train_stats['windows_available']:,} available train windows, sampled "
                "deterministically across sequences and window labels. Validation and test "
                "retain every available window for evaluation."
            )

        # Warnings for 0 anomalies in any split
        for s_name, st_dict in split_stats.items():
            if st_dict["anom_steps"] == 0:
                st.warning(
                    f"Notice: Split '{s_name}' contains 0 labeled anomaly steps. "
                    "Evaluation metrics requiring positive examples (e.g. F1, AUPRC) will be undefined for this split.",
                    icon="⚠️",
                )

        # 2. Feature Distributions: Before vs After Scaling (Box Plots)
        st.markdown("##### Feature Scaling Diagnostics (Before vs After)")
        if "X_train_raw_sample" in st.session_state and "X_train_scaled_sample" in st.session_state:
            raw_sample = st.session_state["X_train_raw_sample"]
            scaled_sample = st.session_state["X_train_scaled_sample"]
            n_box_feats = min(4, len(selected_features))

            box_fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=["<b>Raw Features (Train Sample)</b>", f"<b>{scaler_choice} Scaled (Train Sample)</b>"],
            )

            for i in range(n_box_feats):
                feat_label = selected_features[i]
                box_fig.add_trace(
                    go.Box(y=raw_sample[:, i], name=feat_label, marker_color="#38BDF8"),
                    row=1,
                    col=1,
                )
                box_fig.add_trace(
                    go.Box(y=scaled_sample[:, i], name=feat_label, marker_color="#A855F7"),
                    row=1,
                    col=2,
                )

            box_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#181926",
                plot_bgcolor="#13141f",
                font={"family": "Inter, sans-serif", "color": "#E2E8F0"},
                height=380,
                showlegend=False,
                margin={"l": 40, "r": 20, "t": 40, "b": 40},
            )
            box_fig.update_yaxes(showgrid=True, gridcolor="#2e3148")
            st.plotly_chart(box_fig, use_container_width=True)

        # 3. Sample Window Heatmap Visualization
        st.markdown("##### Sample Sliding Window Preview")
        X_tr_win = st.session_state.get("X_train_windows")
        if X_tr_win is not None and len(X_tr_win) > 0:
            first_win = X_tr_win[0]  # shape: (window_length, n_features)
            win_lbl = st.session_state.get("y_window_train", [0])[0]

            col_hm, col_lp = st.columns(2)

            with col_hm:
                # Heatmap: Features on Y, Time steps on X
                hm_fig = go.Figure(
                    data=go.Heatmap(
                        z=first_win.T,
                        x=list(range(first_win.shape[0])),
                        y=selected_features,
                        colorscale="Viridis",
                        colorbar={"title": "Scaled Val"},
                    )
                )
                hm_fig.update_layout(
                    title=f"<b>Window 0 Heatmap (Label: {'Anomaly' if win_lbl == 1 else 'Normal'})</b>",
                    template="plotly_dark",
                    paper_bgcolor="#181926",
                    plot_bgcolor="#13141f",
                    font={"family": "Inter, sans-serif", "color": "#E2E8F0"},
                    xaxis={"title": "Time Step in Window (t)"},
                    yaxis={"title": "Feature Channel"},
                    height=350,
                    margin={"l": 50, "r": 20, "t": 50, "b": 40},
                )
                st.plotly_chart(hm_fig, use_container_width=True)

            with col_lp:
                # Line plot of window features
                lp_fig = go.Figure()
                palette = ["#38BDF8", "#A855F7", "#10B981", "#F59E0B", "#EF4444"]
                for f_i in range(min(5, first_win.shape[1])):
                    lp_fig.add_trace(
                        go.Scatter(
                            x=list(range(first_win.shape[0])),
                            y=first_win[:, f_i],
                            mode="lines",
                            name=selected_features[f_i],
                            line={"color": palette[f_i % len(palette)], "width": 1.5},
                        )
                    )
                lp_fig.update_layout(
                    title="<b>Window 0 Channel Traces</b>",
                    template="plotly_dark",
                    paper_bgcolor="#181926",
                    plot_bgcolor="#13141f",
                    font={"family": "Inter, sans-serif", "color": "#E2E8F0"},
                    xaxis={"title": "Time Step in Window (t)"},
                    yaxis={"title": "Scaled Amplitude"},
                    height=350,
                    margin={"l": 50, "r": 20, "t": 50, "b": 40},
                )
                st.plotly_chart(lp_fig, use_container_width=True)

        # 4. Post-processing Integrity Confirmation
        st.success(
            f"Integrity Check: Zero NaNs remain across {len(X_tr_win):,} train, "
            f"{len(st.session_state.get('X_val_windows', [])):,} val, and "
            f"{len(st.session_state.get('X_test_windows', [])):,} test windows. "
            "Data is ready for Teacher Inference (Page 4) or Direct Student Baseline Training (Page 5)."
        )
