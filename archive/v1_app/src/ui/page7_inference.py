"""Page 7: Inference, Anomaly Timeline & Deployment Export.

Runs low-latency edge inference on new or test telemetry, aggregates continuous
step scores across overlapping windows, isolates discrete anomaly intervals,
and exports comprehensive reporting artifacts (CSV, ZIP).
"""

from __future__ import annotations

import io
import hashlib
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from src.evaluation.localization import group_intervals
from src.evaluation.metrics import compute_step_metrics
from src.inference.pipeline import prepare_inference_features, run_student_inference
from src.inference.context import robust_context_scores
from src.visualization.plots import (
    plot_anomaly_scores,
    plot_time_series_with_anomalies,
)
from src.visualization.report import (
    create_report_zip,
    metrics_to_csv,
    predictions_to_csv,
)

logger = logging.getLogger(__name__)


def _parse_missing_code_text(value: str) -> list[Any]:
    """Parse comma-separated numeric or literal sensor missing-value codes."""
    parsed: list[Any] = []
    for item in value.split(","):
        token = item.strip()
        if not token:
            continue
        try:
            numeric = float(token)
            parsed.append(int(numeric) if numeric.is_integer() else numeric)
        except ValueError:
            parsed.append(token)
    return parsed


def _read_uploaded_telemetry_csv(file_obj: Any) -> pd.DataFrame:
    """Read comma- or semicolon-delimited sensor CSVs with a header row.

    Decimal commas are common in public environmental datasets and are
    independent of the field delimiter (for example, ``;`` plus ``2,6``).
    """
    return pd.read_csv(file_obj, sep=None, engine="python", decimal=",")


def _timestamps_for_csv(
    dataframe: pd.DataFrame,
    selection: str,
    combined_selection: Optional[str],
    date_column: Optional[str],
    time_column: Optional[str],
) -> np.ndarray:
    """Return row-order, a selected timestamp column, or combined date/time."""
    if selection == "Use row order":
        return np.arange(len(dataframe))
    if (
        combined_selection
        and selection == combined_selection
        and date_column in dataframe.columns
        and time_column in dataframe.columns
    ):
        date_values = dataframe[date_column].astype("string").str.strip()
        time_values = (
            dataframe[time_column]
            .astype("string")
            .str.strip()
            .str.replace(".", ":", regex=False)
        )
        return pd.to_datetime(
            date_values + " " + time_values,
            dayfirst=True,
            errors="coerce",
        ).to_numpy()
    return dataframe[selection].to_numpy()


def _apply_min_len(predictions: np.ndarray) -> np.ndarray:
    """Drop predicted runs shorter than the checkpoint's validation-tuned minimum length."""
    params = st.session_state.get("localization_params") or {}
    min_len = int(params.get("min_len", 1))
    if min_len <= 1:
        return predictions
    from src.evaluation.postprocess import remove_short

    return remove_short(predictions, min_len).astype(int)


def show() -> None:
    """Render the Inference & Deployment page."""
    st.header("Inference, Anomaly Timeline & Deployment Export")
    st.caption(
        "Score a CSV with the trained student, preserve its rows, and export timestep probabilities plus "
        "point, contextual, and collective anomaly labels when the required context is available."
    )

    # Guard: require model or checkpoint
    model = st.session_state.get("trained_model")
    scaler = st.session_state.get("scaler")
    selected_features: List[str] = st.session_state.get("selected_features", [])
    window_length: int = int(st.session_state.get("window_length", 128))
    stride: int = int(st.session_state.get("stride", 32))
    threshold_default: float = float(np.clip(
        st.session_state.get("selected_threshold", 0.5), 0.0, 1.0
    ))
    missing_strategy: str = str(st.session_state.get("missing_strategy", "interpolate"))
    imputation_stats: Dict[str, Any] = st.session_state.get("train_impute_stats", {})
    inference_missing_codes: List[Any] = list(st.session_state.get("missing_codes", []))

    if model is None:
        st.warning(
            "No student model is active in this Streamlit session. If the app was restarted, "
            "go to **5. Student Training (GRU)**, choose the saved checkpoint, and click "
            "**Load & Activate Checkpoint** before returning here."
        )
        return

    active_checkpoint = st.session_state.get("last_checkpoint_path")
    active_model_info = (
        f"Active checkpoint: `{Path(active_checkpoint).name}`; "
        if active_checkpoint else "Active model: current in-memory student; "
    )
    st.caption(
        active_model_info
        + f"{len(selected_features)} input features, {window_length}-step windows, "
        + f"saved step cutoff {threshold_default:.4f}."
    )

    st.markdown("---")
    st.subheader("1. Select Inference Telemetry Input")

    input_mode = st.radio(
        "Choose Input Telemetry Source:",
        [
            "Option A: Use Unseen Test Partition from Current Session",
            "Option B: Upload CSV to Score",
            "Option C: Paste Telemetry Values Manually",
        ],
        horizontal=True,
    )

    inference_df: Optional[pd.DataFrame] = None
    y_true_timeline: Optional[np.ndarray] = None
    timestamps: Optional[np.ndarray] = None
    scaled_inference_override: Optional[np.ndarray] = None
    input_label_col: Optional[str] = None
    context_entity_column: Optional[str] = None
    context_entity_ids: Optional[np.ndarray] = None
    input_basename = Path(str(st.session_state.get("data_filename", "dataset"))).stem
    if input_basename.startswith("benchmark_"):
        input_basename = input_basename[len("benchmark_"):]

    if input_mode == "Option A: Use Unseen Test Partition from Current Session":
        test_meta = st.session_state.get("test_sequence_meta", [])
        if test_meta:
            sequence_options = [
                f"{item.get('group_id', 'sequence')} · {len(item.get('y_step', [])):,} steps · "
                f"{int(np.sum(item.get('y_step', []))):,} anomaly steps"
                for item in test_meta
            ]
            selected_sequence_idx = st.selectbox(
                "Choose a held-out test sequence:",
                range(len(test_meta)),
                format_func=lambda idx: sequence_options[idx],
            )
            selected_sequence = test_meta[selected_sequence_idx]
            inference_df = pd.DataFrame(selected_sequence["values_raw"], columns=selected_features)
            timestamps = selected_sequence.get("timestamps")
            if timestamps is None:
                timestamps = np.arange(len(inference_df))
            y_true_timeline = np.asarray(selected_sequence.get("y_step", []), dtype=int)
            scaled_inference_override = np.asarray(selected_sequence["values_scaled"], dtype=np.float32)
            st.info(
                f"Using held-out sequence '{selected_sequence.get('group_id', 'sequence')}' "
                f"with {len(inference_df):,} timesteps."
            )
        else:
            raw_df = st.session_state.get("raw_df")
            if raw_df is not None and "X_test_windows" in st.session_state:
                split_stats = st.session_state.get("split_stats", {})
                test_T = split_stats.get("Test", {}).get("timesteps", 500)
                inference_df = raw_df.tail(test_T).copy()
                time_col = st.session_state.get("time_col")
                timestamps = inference_df[time_col].values if time_col and time_col in inference_df.columns else np.arange(len(inference_df))
                label_col = st.session_state.get("label_col")
                if label_col and label_col in inference_df.columns:
                    anom_str = str(st.session_state.get("anomaly_label_val", "1")).strip()
                    y_true_timeline = (inference_df[label_col].astype(str).str.strip() == anom_str).astype(int).values
                st.info(f"Using test sequence of length T = {len(inference_df):,} timesteps from session.")
            else:
                st.error("Test partition not found in session state. Please run Step 3 (Preprocessing) first.")
                return

    elif input_mode == "Option B: Upload CSV to Score":
        uploaded_inf_file = st.file_uploader(
            "Upload Target CSV for Anomaly Detection",
            type=["csv", "txt"],
            key="inf_csv_uploader",
        )
        if uploaded_inf_file is not None:
            try:
                # Public telemetry collections often use semicolons as field
                # separators (including AirQualityUCI), so let pandas sniff
                # the delimiter instead of assuming a comma.
                inference_df = _read_uploaded_telemetry_csv(uploaded_inf_file)
                input_basename = Path(uploaded_inf_file.name).stem
                st.success(f"Loaded '{uploaded_inf_file.name}' ({len(inference_df):,} rows).")

                missing_features = [name for name in selected_features if name not in inference_df.columns]
                if missing_features:
                    st.error(
                        "This checkpoint requires these feature columns: "
                        + ", ".join(missing_features)
                    )
                    return

                extra_cols = [name for name in inference_df.columns if name not in selected_features]
                if extra_cols:
                    st.info(
                        "Extra CSV columns will be preserved in the download and ignored by the model: "
                        + ", ".join(extra_cols)
                    )
                train_label_col = st.session_state.get("label_col")
                timestamp_candidates = [
                    name for name in inference_df.columns
                    if name not in selected_features and name != train_label_col
                ]
                date_column = next(
                    (name for name in timestamp_candidates if name.lower() == "date"), None
                )
                time_column = next(
                    (name for name in timestamp_candidates if name.lower() == "time"), None
                )
                combined_time_option = (
                    f"{date_column} + {time_column}"
                    if date_column is not None and time_column is not None
                    else None
                )
                preferred_time_col = st.session_state.get("time_col")
                common_time_cols = {"timestamp", "time", "datetime", "date", "date_time"}
                suggested_time_col = (
                    preferred_time_col if preferred_time_col in timestamp_candidates
                    else combined_time_option
                    if combined_time_option is not None
                    else next((name for name in timestamp_candidates if name.lower() in common_time_cols), None)
                )
                time_options = ["Use row order"]
                if combined_time_option is not None:
                    time_options.append(combined_time_option)
                time_options.extend(timestamp_candidates)
                selected_time_option = st.selectbox(
                    "Timestamp column (optional):",
                    time_options,
                    index=(time_options.index(suggested_time_col) if suggested_time_col in time_options else 0),
                    help="The selected column is preserved in the output and used to label anomaly intervals.",
                )
                timestamps = _timestamps_for_csv(
                    inference_df,
                    selected_time_option,
                    combined_time_option,
                    date_column,
                    time_column,
                )
                if selected_time_option != "Use row order":
                    try:
                        ts_series = pd.Series(timestamps)
                        sortable_ts = (
                            pd.to_numeric(ts_series, errors="coerce")
                            if pd.api.types.is_numeric_dtype(ts_series)
                            else pd.to_datetime(ts_series, errors="coerce")
                        )
                        if sortable_ts.notna().all() and not sortable_ts.is_monotonic_increasing:
                            st.warning(
                                "The selected timestamps are not in ascending order. The GRU and contextual "
                                "checks use CSV row order; sort the file chronologically before inference."
                            )
                    except (TypeError, ValueError):
                        pass

                context_candidates = [
                    name for name in inference_df.columns
                    if name not in selected_features
                    and name != train_label_col
                    and name not in {date_column, time_column}
                    and name != selected_time_option
                ]
                context_choice = st.selectbox(
                    "Entity / account / machine ID for contextual checks (optional):",
                    ["No entity ID"] + context_candidates,
                    help=(
                        "Select a real identifier so each row is compared with that same entity's preceding "
                        "observations. This is required for per-cardholder, per-patient, or per-machine baselines."
                    ),
                    key="inference_context_entity_column",
                )
                if context_choice != "No entity ID":
                    context_entity_column = context_choice
                    context_entity_ids = inference_df[context_choice].to_numpy(dtype=object)
                    st.caption(
                        "Context is computed causally from up to the previous 64 rows for the same ID; "
                        "the first 16 observations per ID are used only to establish a baseline, and rows "
                        "with a missing ID are left context-unscored."
                    )
                train_anomaly_label = str(st.session_state.get("anomaly_label_val", "1")).strip()
                if train_label_col and train_label_col in inference_df.columns:
                    input_label_col = train_label_col
                    y_true_timeline = (
                        inference_df[input_label_col].astype(str).str.strip() == train_anomaly_label
                    ).astype(np.int8).to_numpy()
                    st.caption(
                        f"Ground-truth column `{input_label_col}` is excluded from model input and used only for optional scoring."
                    )
                missing_code_text = st.text_input(
                    "Additional missing-value codes (comma-separated):",
                    value=", ".join(str(code) for code in inference_missing_codes),
                    help="Codes such as -200 are converted to missing before the saved imputation and scaler are applied.",
                )
                inference_missing_codes = _parse_missing_code_text(missing_code_text)

            except Exception as e:
                st.error(f"Failed to parse uploaded CSV: {e}")
                return

    else:
        input_basename = "manual_input"
        st.markdown("Enter comma-separated values (one time step per line, matching selected features):")
        feat_hint = ", ".join(selected_features) if selected_features else "feat_1, feat_2, ..."
        st.caption(f"Expected columns: `{feat_hint}`")

        pasted_text = st.text_area(
            "Telemetry Matrix",
            value="1.0, 2.0, 0.5\n1.2, 2.1, 0.4\n1.1, 2.2, 0.6\n3.5, 9.8, 8.2\n1.0, 2.0, 0.5",
            height=130,
        )
        if pasted_text.strip():
            try:
                inference_df = pd.read_csv(io.StringIO(pasted_text.strip()), header=None)
                if len(selected_features) == inference_df.shape[1]:
                    inference_df.columns = selected_features
                timestamps = np.arange(len(inference_df))
            except Exception as pe:
                st.error(f"Failed to parse manual input: {pe}")
                return

    if inference_df is None or len(inference_df) == 0:
        st.info("Please provide telemetry input to run inference.")
        return

    source_digest = hashlib.sha256()
    source_digest.update(input_mode.encode("utf-8"))
    source_digest.update(pd.util.hash_pandas_object(inference_df, index=True).to_numpy().tobytes())
    source_digest.update(str(list(timestamps) if timestamps is not None else []).encode("utf-8"))
    source_digest.update(str(tuple(map(str, inference_missing_codes))).encode("utf-8"))
    source_digest.update(str(context_entity_column or "no_context_entity").encode("utf-8"))
    source_digest.update(str((tuple(selected_features), window_length, stride, id(model))).encode("utf-8"))
    source_signature = source_digest.hexdigest()

    st.markdown("---")
    st.subheader("2. Operational Threshold Control")

    th_col1, th_col2 = st.columns([1, 2])
    with th_col1:
        inference_threshold = st.slider(
            "Anomaly Score Threshold:",
            min_value=0.0,
            max_value=1.0,
            value=threshold_default,
            step=0.0001,
            format="%.4f",
            help=(
                "Continuous student score cutoff. The default is the saved validation cutoff; "
                "lowering it increases alerts and may increase false positives."
            ),
        )

    with th_col2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        run_inf_btn = st.button("Execute Model Inference & Scoring", type="primary", use_container_width=True)

    calibrated_context_threshold = float(
        st.session_state.get("contextual_threshold_robust_z", 4.5)
    )
    context_threshold = calibrated_context_threshold
    if context_entity_ids is not None:
        context_widget_key = "context_robust_z_threshold"
        if context_widget_key not in st.session_state:
            st.session_state[context_widget_key] = min(
                20.0, max(2.0, calibrated_context_threshold)
            )
        context_threshold = st.slider(
            "Contextual robust-z threshold:",
            min_value=2.0,
            max_value=20.0,
            step=0.5,
            help=(
                "A row is contextually unusual when its largest feature deviation exceeds this many "
                "robust standard deviations from that entity's trailing median. The starting value is "
                "calibrated on normal validation rows to a 0.5% target false-alarm rate."
            ),
            key=context_widget_key,
        )
        calibration_rows = int(
            st.session_state.get("contextual_calibration_normal_rows", 0)
        )
        if calibration_rows:
            st.caption(
                f"Validation-calibrated robust-z starting cutoff: {calibrated_context_threshold:.2f} "
                f"({calibration_rows:,} sampled normal rows; target 0.5% false alarms, "
                "with a conservative minimum of 4.5)."
            )
        else:
            st.caption(
                f"No saved validation calibration is available for this older checkpoint; using the "
                f"conservative robust-z default {calibrated_context_threshold:.2f}. "
                "Re-run preprocessing and train a new checkpoint to calibrate this cutoff."
            )

    creditcard_features = {f"V{i}" for i in range(1, 29)} | {"Amount"}
    independent_observations = creditcard_features.issubset(set(inference_df.columns))
    if independent_observations and context_entity_ids is None:
        st.info(
            "CreditCard rows are treated as independent transactions. This CSV has no cardholder ID, "
            "so the student detects fraud-like multivariate patterns; it cannot learn what is unusually "
            "large for one person's spending history."
        )

    # Execute Inference Pipeline
    if run_inf_btn:
        with st.spinner("Processing telemetry through student model..."):
            try:
                feat_cols = list(selected_features)
                if scaled_inference_override is not None:
                    X_raw_inf = inference_df.loc[:, feat_cols].apply(
                        pd.to_numeric, errors="coerce"
                    ).to_numpy(dtype=np.float32)
                    X_scaled_inf = scaled_inference_override
                else:
                    X_raw_inf, X_scaled_inf = prepare_inference_features(
                        dataframe=inference_df,
                        feature_names=selected_features,
                        scaler=scaler,
                        missing_strategy=missing_strategy,
                        imputation_stats=imputation_stats,
                        missing_codes=inference_missing_codes,
                    )

                # The helper materializes only each small inference batch, not
                # every overlapping window in memory at once.
                agg_scores, inf_time_sec = run_student_inference(
                    model=model,
                    scaled_values=X_scaled_inf,
                    window_length=window_length,
                    stride=stride,
                    batch_size=64,
                    localization=st.session_state.get("localization_params"),
                )
                tot_steps = len(X_scaled_inf)

                context_started = time.perf_counter()
                if context_entity_ids is not None:
                    contextual_scores, contextual_drivers, contextual_available = robust_context_scores(
                        values=X_scaled_inf,
                        entity_ids=context_entity_ids,
                        feature_names=feat_cols,
                    )
                else:
                    contextual_scores = np.full(tot_steps, np.nan, dtype=np.float32)
                    contextual_drivers = np.full(tot_steps, "", dtype=object)
                    contextual_available = np.zeros(tot_steps, dtype=bool)
                contextual_elapsed_sec = time.perf_counter() - context_started
                total_inference_sec = inf_time_sec + contextual_elapsed_sec

                # 5. Thresholding & Event Interval Clustering
                student_predictions = _apply_min_len((agg_scores >= inference_threshold).astype(int))
                contextual_predictions = (
                    contextual_available & (contextual_scores >= context_threshold)
                ).astype(int)
                y_pred_timeline = np.maximum(student_predictions, contextual_predictions)
                intervals = group_intervals(
                    y_pred_timeline,
                    timestamps=timestamps,
                    independent_observations=independent_observations,
                )

                # Compute Edge Latency & Throughput
                latency_per_sample_ms = (total_inference_sec / max(1, tot_steps)) * 1000.0
                throughput_samples_sec = tot_steps / max(1e-6, total_inference_sec)

                # Store in session_state
                st.session_state["inf_agg_scores"] = agg_scores
                st.session_state["inf_contextual_scores"] = contextual_scores
                st.session_state["inf_contextual_drivers"] = contextual_drivers
                st.session_state["inf_contextual_available"] = contextual_available
                st.session_state["inf_context_entity_column"] = context_entity_column
                st.session_state["inf_context_entity_ids"] = context_entity_ids
                st.session_state["inf_independent_observations"] = independent_observations
                st.session_state["inf_y_pred"] = y_pred_timeline
                st.session_state["inf_intervals"] = intervals
                st.session_state["inf_timestamps"] = timestamps
                st.session_state["inf_X_raw"] = X_raw_inf
                st.session_state["inf_original_df"] = inference_df.copy()
                st.session_state["inf_y_true"] = (
                    None if y_true_timeline is None else np.asarray(y_true_timeline, dtype=np.int8)
                )
                st.session_state["inf_label_col"] = input_label_col
                st.session_state["inf_latency_ms"] = latency_per_sample_ms
                st.session_state["inf_student_seconds"] = inf_time_sec
                st.session_state["inf_context_seconds"] = contextual_elapsed_sec
                st.session_state["inf_total_seconds"] = total_inference_sec
                st.session_state["inf_throughput"] = throughput_samples_sec
                st.session_state["inf_feat_cols"] = feat_cols
                st.session_state["inf_source_signature"] = source_signature
                st.session_state["inference_complete"] = True

                st.success(
                    f"Student inference and optional contextual checks completed in "
                    f"{total_inference_sec * 1000:.2f} ms ({tot_steps:,} timesteps)!"
                )

            except Exception as e:
                st.error(f"Inference pipeline encountered an error: {e}")
                logger.exception("Inference execution failure")
                return

    # Results & Visualization Section
    if (
        st.session_state.get("inference_complete", False)
        and st.session_state.get("inf_source_signature") == source_signature
    ):
        st.markdown("---")
        st.subheader("3. Anomaly Timeline & Diagnostic Visualizations")

        agg_scores = st.session_state["inf_agg_scores"]
        ts_timeline = st.session_state["inf_timestamps"]
        y_true_timeline = st.session_state.get("inf_y_true")
        contextual_scores = np.asarray(
            st.session_state.get("inf_contextual_scores", np.full(len(agg_scores), np.nan)),
            dtype=float,
        )
        contextual_drivers = np.asarray(
            st.session_state.get("inf_contextual_drivers", np.full(len(agg_scores), "")),
            dtype=object,
        )
        contextual_available = np.asarray(
            st.session_state.get("inf_contextual_available", np.zeros(len(agg_scores))),
            dtype=bool,
        )
        context_entity_column = st.session_state.get("inf_context_entity_column")
        independent_observations = bool(
            st.session_state.get("inf_independent_observations", False)
        )
        # Threshold changes take effect immediately without re-running the GRU.
        student_predictions = _apply_min_len((agg_scores >= inference_threshold).astype(int))
        contextual_predictions = (
            contextual_available & (contextual_scores >= context_threshold)
        ).astype(int)
        y_pred_timeline = np.maximum(student_predictions, contextual_predictions)
        intervals = group_intervals(
            y_pred_timeline,
            timestamps=ts_timeline,
            independent_observations=independent_observations,
        )
        st.session_state["inf_y_pred"] = y_pred_timeline
        st.session_state["inf_intervals"] = intervals
        X_raw_inf = st.session_state["inf_X_raw"]
        latency_ms = st.session_state["inf_latency_ms"]
        contextual_elapsed_sec = float(st.session_state.get("inf_context_seconds", 0.0))
        throughput_rate = st.session_state["inf_throughput"]
        feat_cols = st.session_state["inf_feat_cols"]

        csv_original_df = st.session_state.get("inf_original_df")
        if csv_original_df is not None and y_true_timeline is not None:
            csv_original_df = csv_original_df.copy()
            csv_original_df["ground_truth_is_anomaly"] = np.asarray(y_true_timeline, dtype=np.int8)

        preds_csv = predictions_to_csv(
            timestamps=ts_timeline,
            scores=agg_scores,
            predictions=y_pred_timeline,
            feature_df=pd.DataFrame(X_raw_inf, columns=feat_cols),
            original_df=csv_original_df,
            student_predictions=student_predictions,
            contextual_scores=contextual_scores,
            contextual_predictions=contextual_predictions,
            contextual_drivers=contextual_drivers,
            contextual_available=contextual_available,
            student_threshold=inference_threshold,
            contextual_threshold=(context_threshold if context_entity_column else None),
            context_entity_column=context_entity_column,
            independent_observations=independent_observations,
        )

        safe_stem = "".join(
            char if char.isalnum() or char in "-_" else "_"
            for char in input_basename
        ).strip("_") or "dataset"
        output_path = Path("report") / f"{safe_stem}_predictions.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(preds_csv, encoding="utf-8", newline="")

        st.markdown("#### Predictions CSV")
        st.download_button(
            label=f"📥 Download {safe_stem} Predictions CSV ({len(agg_scores):,} rows)",
            data=preds_csv,
            file_name=output_path.name,
            mime="text/csv",
            use_container_width=True,
            on_click="ignore",
            key="predictions_csv_download_top",
        )
        st.caption(
            f"A copy is saved in `{output_path.resolve()}`. The CSV contains every timestep; "
            "the charts below are sampled for display speed. Student and contextual evidence are exported separately."
        )

        max_chart_points = 6_000
        if len(agg_scores) > max_chart_points:
            chart_indices = np.linspace(
                0, len(agg_scores) - 1, num=max_chart_points, dtype=np.int64
            )
        else:
            chart_indices = np.arange(len(agg_scores), dtype=np.int64)
        chart_values = X_raw_inf[chart_indices]
        chart_timestamps = (
            np.asarray(ts_timeline)[chart_indices]
            if ts_timeline is not None
            else chart_indices
        )
        true_intervals = (
            group_intervals(y_true_timeline, timestamps=ts_timeline)
            if y_true_timeline is not None
            else None
        )

        # Latency & Throughput Banner
        l_col1, l_col2, l_col3, l_col4 = st.columns(4)
        with l_col1:
            st.metric("Total Timesteps Scored", f"{len(agg_scores):,}")
        with l_col2:
            st.metric("Detected Anomaly Events", f"{len(intervals):,}")
        with l_col3:
            st.metric("Edge Latency", f"{latency_ms:.3f} ms / step")
        with l_col4:
            st.metric("Throughput", f"{throughput_rate:,.1f} steps / sec")

        score_quantiles = np.quantile(agg_scores, [0.50, 0.95, 0.99, 1.00])
        score_cols = st.columns(5)
        for col, label, value in zip(
            score_cols[:4],
            ("Score median", "Score 95th pct.", "Score 99th pct.", "Max student score"),
            score_quantiles,
        ):
            col.metric(label, f"{float(value):.5f}")
        score_cols[4].metric(
            "Student points above cutoff", f"{int(student_predictions.sum()):,}"
        )
        if np.ptp(agg_scores) <= 1e-6:
            st.warning(
                "The student is returning nearly constant scores. The current cutoff may produce "
                "zero or all-point flags, but it cannot reliably separate anomalies; check the "
                "active checkpoint and input feature values."
            )
        elif int(student_predictions.sum()) == 0:
            max_student_score = float(np.max(agg_scores))
            if max_student_score < inference_threshold:
                st.warning(
                    f"No student scores reach the selected cutoff: max score is "
                    f"{max_student_score:.5f}, while the cutoff is {inference_threshold:.5f}. "
                    "Zero flags are therefore expected at this setting. Confirm that this is the "
                    "checkpoint trained for these feature columns; if you have labels, recalibrate "
                    "the cutoff on validation data. Lowering it by eye can increase false alarms."
                )
        if context_entity_column:
            st.caption(
                f"Contextual alerts: {int(contextual_predictions.sum()):,} rows using `{context_entity_column}`; "
                f"rows without 16 prior observations for that entity are left unscored by the contextual check. "
                f"Context scoring took {contextual_elapsed_sec:.3f} s."
            )

        # 1. Multi-Channel Signal Overlay with Anomaly Shading
        st.markdown("##### Multi-Channel Telemetry with Detected Anomaly Intervals (Red)")
        timeline_fig = plot_time_series_with_anomalies(
            values=chart_values,
            timestamps=chart_timestamps,
            feature_names=feat_cols,
            ground_truth_intervals=true_intervals,
            anomaly_intervals=intervals,
            title="Telemetry Signals & Detected Anomaly Spans",
        )
        st.plotly_chart(timeline_fig, use_container_width=True)

        # 2. Continuous Anomaly Score Curve
        st.markdown("##### Continuous Anomaly Probability Timeline")
        st.caption(
            "The line is the GRU student's probability. Red shading shows the final anomaly flag, "
            "which also includes contextual alerts when an entity ID was selected."
        )
        score_fig = plot_anomaly_scores(
            step_scores=agg_scores[chart_indices],
            timestamps=chart_timestamps,
            threshold=inference_threshold,
            ground_truth_intervals=true_intervals,
            anomaly_intervals=intervals,
            title="Step-Level Anomaly Probabilities & Decision Threshold",
        )
        st.plotly_chart(score_fig, use_container_width=True)

        # 3. Detected Anomaly Event Intervals Table
        st.markdown("##### Detected Anomaly Events Table")
        if intervals:
            int_rows = []
            for idx, iv in enumerate(intervals):
                s_idx = iv["start_idx"]
                e_idx = iv["end_idx"]
                peak_score = float(np.max(agg_scores[s_idx : e_idx + 1]))
                int_rows.append({
                    "Event #": idx + 1,
                    "Event Type": (
                        "group_anomaly"
                        if not independent_observations and iv["length"] > 1
                        else "contextual_anomaly"
                        if np.any(contextual_predictions[s_idx : e_idx + 1])
                        else "point_anomaly"
                    ),
                    "Anomaly Taxonomy": (
                        "collective_anomaly"
                        if not independent_observations and iv["length"] > 1
                        else "contextual_anomaly"
                        if np.any(contextual_predictions[s_idx : e_idx + 1])
                        else "point_anomaly"
                    ),
                    "Detection Basis": (
                        "student+context"
                        if np.any(student_predictions[s_idx : e_idx + 1])
                        and np.any(contextual_predictions[s_idx : e_idx + 1])
                        else "contextual"
                        if np.any(contextual_predictions[s_idx : e_idx + 1])
                        else "student"
                    ),
                    "Start Index": s_idx,
                    "End Index": e_idx,
                    "Start Timestamp": str(iv["start_ts"]) if iv["start_ts"] is not None else str(s_idx),
                    "End Timestamp": str(iv["end_ts"]) if iv["end_ts"] is not None else str(e_idx),
                    "Duration (Steps)": iv["length"],
                    "Peak Anomaly Score": f"{peak_score:.4f}",
                })
            st.dataframe(pd.DataFrame(int_rows), use_container_width=True)
        else:
            st.info("No anomalies detected above the current decision threshold.")

        # 4. Report Generation & Download Section
        st.markdown("---")
        st.subheader("4. Artifact & Report Export")

        metrics_csv_str = ""
        metrics_available = False
        if y_true_timeline is not None and len(y_true_timeline) == len(agg_scores):
            inference_metrics = compute_step_metrics(
                np.asarray(y_true_timeline, dtype=np.int8),
                agg_scores,
                threshold=inference_threshold,
            )
            metrics_csv_str = metrics_to_csv({"evaluation_source": "current inference CSV", **inference_metrics})
            metrics_available = True
            st.markdown("##### Scores Against Labels Present in This CSV")
            st.caption(
                "These metrics evaluate the student probability at the selected student threshold; "
                "optional contextual alerts are reported separately in the predictions CSV."
            )
            metric_cols = st.columns(5)
            for col, name in zip(metric_cols, ("precision", "recall", "f1", "pr_auc", "roc_auc")):
                value = inference_metrics.get(name, float("nan"))
                col.metric(name.upper(), f"{value:.4f}" if np.isfinite(value) else "N/A")
        else:
            heldout_step_metrics = st.session_state.get("test_step_metrics")
            if heldout_step_metrics is not None:
                metrics_csv_str = metrics_to_csv({
                    "evaluation_source": "held-out test set (not this unlabeled CSV)",
                    **heldout_step_metrics.to_dict(),
                })
                metrics_available = True
            st.info(
                "This input has no usable ground-truth labels. The CSV contains model scores and anomaly marks; "
                "accuracy metrics are reported separately for the held-out test set, if available."
            )

        # Generate full ZIP archive
        zip_bytes = create_report_zip(
            checkpoint_path=st.session_state.get("last_checkpoint_path"),
            metrics_csv=metrics_csv_str if metrics_csv_str else None,
            predictions_csv=preds_csv,
            metadata_json={
                "threshold": inference_threshold,
                "latency_ms_per_sample": latency_ms,
                "throughput_samples_per_sec": throughput_rate,
                "contextual_scoring_seconds": contextual_elapsed_sec,
                "feature_names": feat_cols,
                "missing_codes": inference_missing_codes,
                "context_entity_column": context_entity_column,
                "contextual_robust_z_threshold": (
                    context_threshold if context_entity_column else None
                ),
                "n_anomaly_events": len(intervals),
                "independent_observations": independent_observations,
                "anomaly_type_rule": (
                    "anomaly_type preserves point_anomaly/group_anomaly; anomaly_taxonomy maps group_anomaly "
                    "to collective_anomaly. Independent transactions stay separate; contextual alerts are "
                    "contextual_anomaly."
                ),
                "metrics_source": "current inference CSV" if y_true_timeline is not None else "held-out test set or unavailable",
            },
        )

        d_col1, d_col2 = st.columns(2)

        with d_col1:
            st.download_button(
                label="📥 Download Metrics CSV",
                data=metrics_csv_str if metrics_csv_str else "No metrics recorded",
                file_name="evaluation_metrics.csv",
                mime="text/csv",
                use_container_width=True,
                disabled=not metrics_available,
            )

        with d_col2:
            st.download_button(
                label="📦 Download Full Experiment ZIP Archive",
                data=zip_bytes,
                file_name="experiment_report_archive.zip",
                mime="application/zip",
                use_container_width=True,
            )
