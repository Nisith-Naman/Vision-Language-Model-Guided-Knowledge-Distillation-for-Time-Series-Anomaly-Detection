"""Page 2: Upload & Inspect Data.

Enables loading data via custom CSV upload, pre-configured benchmark datasets,
or synthetic time-series generation, with automated data profiling and feature mapping.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yaml

from src.data.loader import DataQualityReport, Series, load_csv, load_dataset


def generate_synthetic_data(
    n_samples: int = 2000,
    n_features: int = 3,
    anomaly_rate: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic synthetic multidimensional time-series data with injected anomalies.

    Creates baseline seasonal dynamics, autoregressive noise, injected point spikes,
    level shifts, and noise bursts. Labeled explicitly as synthetic data.

    Args:
        n_samples: Total number of time steps (T).
        n_features: Number of numerical sensor features (F).
        anomaly_rate: Targeted fraction of anomalous time steps.
        seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame with timestamp, sensor features, and ground-truth anomaly label.
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2025-01-01 00:00:00", periods=n_samples, freq="1min")

    time_idx = np.arange(n_samples)
    values = np.zeros((n_samples, n_features), dtype=np.float32)
    labels = np.zeros(n_samples, dtype=np.int8)

    feature_cols = [f"sensor_{i + 1}" for i in range(n_features)]

    # 1. Generate base continuous signals (harmonics + random walk drift)
    for f in range(n_features):
        period_1 = rng.uniform(40, 120)
        period_2 = rng.uniform(150, 300)
        phase_1 = rng.uniform(0, 2 * np.pi)
        phase_2 = rng.uniform(0, 2 * np.pi)

        harmonic = (
            np.sin(2 * np.pi * time_idx / period_1 + phase_1)
            + 0.5 * np.cos(2 * np.pi * time_idx / period_2 + phase_2)
        )
        # AR(1) autoregressive noise
        noise = np.zeros(n_samples, dtype=np.float32)
        ar_coeff = 0.7
        for t in range(1, n_samples):
            noise[t] = ar_coeff * noise[t - 1] + rng.normal(0, 0.25)

        baseline = 10.0 * (f + 1) + harmonic + noise
        values[:, f] = baseline

    # 2. Inject point anomalies (isolated extreme spikes)
    n_anomalies_target = int(n_samples * anomaly_rate)
    n_point_spikes = max(2, int(n_anomalies_target * 0.4))
    spike_indices = rng.choice(n_samples, size=n_point_spikes, replace=False)

    for idx in spike_indices:
        feat_to_perturb = rng.choice(n_features)
        spike_magnitude = rng.choice([-1, 1]) * rng.uniform(4.0, 7.5) * np.std(values[:, feat_to_perturb])
        values[idx, feat_to_perturb] += spike_magnitude
        labels[idx] = 1

    # 3. Inject level shifts (sustained baseline offset for 15-40 steps)
    remaining_budget = max(0, n_anomalies_target - int(np.sum(labels == 1)))
    n_shifts = max(1, remaining_budget // 25)

    for _ in range(n_shifts):
        shift_len = int(rng.integers(15, 35))
        if n_samples - shift_len - 10 <= 10:
            break
        start_idx = int(rng.integers(10, n_samples - shift_len - 10))
        end_idx = start_idx + shift_len

        feat_to_shift = rng.choice(n_features)
        shift_val = rng.choice([-1, 1]) * rng.uniform(3.0, 5.5) * np.std(values[:, feat_to_shift])
        values[start_idx:end_idx, feat_to_shift] += shift_val
        labels[start_idx:end_idx] = 1

    # 4. Inject noise burst anomalies (sustained variance flare-up)
    if remaining_budget > 20:
        burst_len = int(rng.integers(15, 30))
        start_idx = int(rng.integers(10, n_samples - burst_len - 10))
        end_idx = start_idx + burst_len
        values[start_idx:end_idx, :] += rng.normal(0, 2.5, size=(burst_len, n_features))
        labels[start_idx:end_idx] = 1

    df = pd.DataFrame(values, columns=feature_cols)
    df.insert(0, "timestamp", timestamps)
    df["label"] = labels

    return df


def _build_data_quality_report(df: pd.DataFrame, filename: str) -> DataQualityReport:
    """Construct a DataQualityReport dataclass instance from a DataFrame."""
    n_rows, n_cols = df.shape
    dtypes_dict = {col: str(dtype) for col, dtype in df.dtypes.items()}
    missing_counts = df.isna().sum().to_dict()
    duplicate_count = int(df.duplicated().sum())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    sample_rows = df.head(5).copy()

    return DataQualityReport(
        filename=filename,
        n_rows=n_rows,
        n_cols=n_cols,
        dtypes_dict=dtypes_dict,
        missing_counts_dict=missing_counts,
        duplicate_count=duplicate_count,
        numeric_cols=numeric_cols,
        non_numeric_cols=non_numeric_cols,
        sample_rows=sample_rows,
    )


def _benchmark_to_frame(series_list: List[Series], max_rows: int = 50_000) -> pd.DataFrame:
    """Build a bounded UI preview; preserve the complete series list for training."""
    total_rows = sum(series.n_timesteps for series in series_list)
    frames: List[pd.DataFrame] = []
    remaining = max_rows
    for idx, series in enumerate(series_list):
        if remaining <= 0:
            break
        quota = min(series.n_timesteps, max(1, int(round(max_rows * series.n_timesteps / max(total_rows, 1)))), remaining)
        row_indices = np.linspace(0, series.n_timesteps - 1, num=quota, dtype=np.int64) if quota < series.n_timesteps else np.arange(series.n_timesteps, dtype=np.int64)
        frame = pd.DataFrame(series.values[row_indices], columns=series.feature_names)
        if series.timestamps is not None:
            frame.insert(0, "timestamp", series.timestamps[row_indices])
        frame["label"] = series.y_step[row_indices] if series.y_step is not None else np.zeros(len(row_indices), dtype=np.int8)
        frame["_series_id"] = f"sequence_{idx:05d}"
        frame["_split_group"] = str(
            series.metadata.get("subject_id")
            or series.metadata.get("instance_file")
            or series.group_id
        )
        frame["_split_role"] = str(series.metadata.get("split_role", ""))
        frames.append(frame)
        remaining -= len(frame)
    preview = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    preview.attrs["full_timesteps"] = total_rows
    preview.attrs["is_sampled_preview"] = total_rows > len(preview)
    return preview


def show() -> None:
    """Render the Upload & Inspect Data page."""
    st.header("Upload & Inspect Data")
    st.caption("Load multi-sensor telemetry, inspect profiling metrics, and map feature columns.")

    data_source = st.radio(
        "Choose Data Source Mode:",
        ["Option 1: Upload Custom CSV", "Option 2: Use Built-in Dataset", "Option 3: Generate Synthetic Data for Demo"],
        horizontal=True,
    )

    loaded_df: Optional[pd.DataFrame] = None
    source_filename = ""
    is_synthetic = False
    benchmark_series_list: Optional[List[Series]] = None
    benchmark_key: Optional[str] = None
    app_config: Dict[str, Any] = {}

    # Option 1: File Uploader
    if data_source == "Option 1: Upload Custom CSV":
        uploaded_file = st.file_uploader(
            "Upload time-series CSV file",
            type=["csv", "txt"],
            help="Upload comma, semicolon, or tab-delimited tabular time series.",
        )
        if uploaded_file is not None:
            source_filename = uploaded_file.name
            try:
                # Save temporarily to cache to allow separator sniffing via load_csv
                cache_dir = Path("./cache")
                cache_dir.mkdir(parents=True, exist_ok=True)
                temp_path = cache_dir / f"upload_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                loaded_df, report = load_csv(temp_path)
                st.success(f"Successfully loaded '{uploaded_file.name}' with separator sniffing.")
            except Exception as e:
                st.error(f"Failed to load uploaded CSV: {e}")
                # Fallback to direct pandas read
                try:
                    uploaded_file.seek(0)
                    loaded_df = pd.read_csv(uploaded_file)
                    report = _build_data_quality_report(loaded_df, uploaded_file.name)
                except Exception as ex2:
                    st.error(f"Direct pandas parsing also failed: {ex2}")
                    return

    # Option 2: Built-in Benchmark Datasets
    elif data_source == "Option 2: Use Built-in Dataset":
        col_ds, col_load_btn = st.columns([3, 1])
        with col_ds:
            ds_choice = st.selectbox(
                "Select Benchmark Dataset:",
                ["creditcard", "daphnet", "airquality", "oil3w", "metropt"],
                help="Pre-configured datasets with domain-specific properties.",
                key="benchmark_dataset_selector",
            )

        # Load configuration
        config_path = Path("config.yaml")
        app_config: Dict[str, Any] = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                app_config = yaml.safe_load(f)

        dataset_configs = app_config.get("datasets", {})
        selected_cfg = dataset_configs.get(ds_choice, {})

        with col_load_btn:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            load_benchmark = st.button("Load Dataset", type="primary", use_container_width=True)

        dataset_already_loaded = (
            st.session_state.get("current_dataset_key") == ds_choice
            and st.session_state.get("pending_benchmark_key") == ds_choice
            and st.session_state.get("pending_benchmark_series_list") is not None
        )
        if load_benchmark or dataset_already_loaded or (
            st.session_state.get("current_dataset_key") == ds_choice and "raw_df" in st.session_state
        ):
            source_filename = f"benchmark_{ds_choice}"
            # Check if dataset files exist on disk
            try:
                cached_series = st.session_state.get("pending_benchmark_series_list")
                cached_preview = st.session_state.get("pending_benchmark_preview_df")
                if (
                    st.session_state.get("current_dataset_key") == ds_choice
                    and st.session_state.get("pending_benchmark_key") == ds_choice
                    and cached_series is not None
                    and cached_preview is not None
                ):
                    series_list = cached_series
                    loaded_df = cached_preview
                else:
                    series_list = load_dataset(ds_choice, selected_cfg)
                    loaded_df = _benchmark_to_frame(series_list)
                    st.session_state["pending_benchmark_key"] = ds_choice
                    st.session_state["pending_benchmark_series_list"] = series_list
                    st.session_state["pending_benchmark_preview_df"] = loaded_df
                if series_list:
                    benchmark_series_list = series_list
                    benchmark_key = ds_choice
                    st.session_state["current_dataset_key"] = ds_choice
                    st.success(
                        f"Loaded all {len(series_list):,} sequence(s) from '{ds_choice}' "
                        f"({loaded_df.attrs['full_timesteps']:,} timesteps)."
                    )
                    if loaded_df.attrs.get("is_sampled_preview", False):
                        st.caption(
                            f"The table and feature plots use a {len(loaded_df):,}-row preview. "
                            "All loaded sequence data remains available for preprocessing and training."
                        )
            except Exception as err:
                st.warning(
                    f"Dataset raw files for '{ds_choice}' not found locally ({err}). "
                    "You can generate synthetic demo data or upload a custom CSV."
                )

    # Option 3: Synthetic Data Generator
    else:
        col_s1, col_s2, col_s3, col_btn = st.columns([1.2, 1.2, 1.2, 1.5])
        with col_s1:
            n_samples_input = st.number_input("Time Steps (T)", min_value=200, max_value=20000, value=2000, step=200)
        with col_s2:
            n_features_input = st.number_input("Sensor Channels (F)", min_value=1, max_value=20, value=3, step=1)
        with col_s3:
            anom_rate_input = st.slider("Anomaly Rate (%)", min_value=1.0, max_value=20.0, value=5.0, step=0.5) / 100.0
        with col_btn:
            st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
            gen_btn = st.button("Generate Synthetic Dataset", type="primary", use_container_width=True)

        if gen_btn or ("raw_df" in st.session_state and st.session_state.get("is_synthetic", False)):
            loaded_df = generate_synthetic_data(
                n_samples=int(n_samples_input),
                n_features=int(n_features_input),
                anomaly_rate=float(anom_rate_input),
            )
            source_filename = "synthetic_multivariate_timeseries.csv"
            is_synthetic = True
            st.info("Generated synthetic benchmark time series with injected spikes, shifts, and noise bursts.")

    # Retain previously loaded DataFrame in session state if user navigated back
    if loaded_df is None and "raw_df" in st.session_state:
        loaded_df = st.session_state["raw_df"]
        source_filename = st.session_state.get("data_filename", "session_data.csv")
        is_synthetic = st.session_state.get("is_synthetic", False)

    if loaded_df is None:
        st.info("Please upload a CSV, select a benchmark dataset, or generate synthetic data to proceed.")
        return

    # Prominent Synthetic Banner if applicable
    if is_synthetic or "synthetic" in source_filename.lower():
        st.warning(
            "SYNTHETIC DATA MODE ACTIVE — The loaded time series consists of synthetic harmonics with "
            "programmatically injected point anomalies and level shifts for testing and demonstration.",
            icon="⚠️",
        )

    # Profile Data Quality
    report = _build_data_quality_report(loaded_df, source_filename)
    preview_is_sampled = bool(loaded_df.attrs.get("is_sampled_preview", False))
    loaded_timesteps = int(loaded_df.attrs.get("full_timesteps", len(loaded_df)))

    if benchmark_key == "airquality" and st.session_state.get("current_dataset_key") == "airquality":
        st.warning(
            "Air Quality has no native anomaly labels. The selected semi-synthetic strategy adds "
            "labeled anomalies independently to train, validation, and test after the split. "
            "Its scores therefore measure detection of injected patterns, not naturally labeled events."
        )

    st.markdown("---")
    st.subheader("Data Quality Profiling Report")

    # Metrics Summary Cards
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    with m1:
        st.metric("Source File", report.filename)
    with m2:
        st.metric("Loaded Timesteps (T)", f"{loaded_timesteps:,}")
    with m3:
        st.metric("Rows Profiled", f"{report.n_rows:,}")
    with m4:
        st.metric("Total Columns", f"{report.n_cols}")
    with m5:
        total_missing = sum(report.missing_counts_dict.values())
        st.metric("Missing Values", f"{total_missing:,}")
    with m6:
        st.metric("Duplicate Rows", f"{report.duplicate_count:,}")
    if preview_is_sampled:
        st.caption("Missing and duplicate counts above are calculated on the bounded preview sample.")

    # First 5 Rows Preview
    with st.expander("Preview First 5 Rows", expanded=True):
        st.dataframe(report.sample_rows, use_container_width=True)

    # Missing Value Breakdown with Visual Color Coding
    col_missing, col_dtypes = st.columns([1, 1])

    with col_missing:
        st.markdown("##### Missing Value Analysis")
        missing_df = pd.DataFrame({
            "Column": list(report.missing_counts_dict.keys()),
            "Missing Count": list(report.missing_counts_dict.values()),
            "Missing %": [round((c / report.n_rows) * 100, 2) for c in report.missing_counts_dict.values()],
        })

        def color_missing_pct(val: float) -> str:
            if val == 0:
                return "background-color: rgba(16, 185, 129, 0.25); color: #10B981;"
            elif val < 10.0:
                return "background-color: rgba(245, 158, 11, 0.25); color: #F59E0B;"
            else:
                return "background-color: rgba(239, 68, 68, 0.25); color: #EF4444;"

        styled_missing = missing_df.style.map(color_missing_pct, subset=["Missing %"])
        st.dataframe(styled_missing, use_container_width=True, height=220)

    with col_dtypes:
        st.markdown("##### Column Data Types")
        dtypes_df = pd.DataFrame({
            "Column": list(report.dtypes_dict.keys()),
            "Data Type": list(report.dtypes_dict.values()),
            "Numeric?": ["Yes" if c in report.numeric_cols else "No" for c in report.dtypes_dict.keys()],
        })
        st.dataframe(dtypes_df, use_container_width=True, height=220)

    st.markdown("---")
    st.subheader("Schema Mapping & Feature Selection")

    # Column Selectors Form
    numeric_columns = report.numeric_cols
    all_columns = loaded_df.columns.tolist()

    c_sel1, c_sel2 = st.columns(2)

    with c_sel1:
        # Timestamp column
        time_options = ["Use row index"] + all_columns
        default_time_idx = 0
        for idx, col in enumerate(time_options):
            if any(k in col.lower() for k in ["time", "date", "timestamp"]):
                default_time_idx = idx
                break
        selected_time_col = st.selectbox(
            "Timestamp Column:",
            options=time_options,
            index=default_time_idx,
            help="Select temporal index or choose 'Use row index' for sequential timesteps.",
        )

        # Label column (optional)
        label_options = ["None (Unsupervised / No Ground Truth)"] + all_columns
        default_label_idx = 0
        for idx, col in enumerate(label_options):
            if any(k in col.lower() for k in ["label", "class", "target", "annotation", "anomaly"]):
                default_label_idx = idx
                break
        selected_label_col = st.selectbox(
            "Ground-Truth Anomaly Label Column (Optional):",
            options=label_options,
            index=default_label_idx,
            help="Optional binary label column for evaluation and benchmarking.",
        )

        # Entity / Series ID column (optional)
        entity_options = ["None (Single Continuous Sequence)"] + all_columns
        default_entity_idx = entity_options.index("_series_id") if (
            benchmark_series_list is not None and "_series_id" in entity_options
        ) else 0
        selected_entity_col = st.selectbox(
            "Entity / Group ID Column (Optional):",
            options=entity_options,
            index=default_entity_idx,
            help="For multi-machine, multi-patient, or multi-well telemetry.",
        )

    with c_sel2:
        # Default features: all numeric columns except time and label
        excluded_from_feats = set()
        if selected_time_col != "Use row index":
            excluded_from_feats.add(selected_time_col)
        if selected_label_col != "None (Unsupervised / No Ground Truth)":
            excluded_from_feats.add(selected_label_col)
        if selected_entity_col != "None (Single Continuous Sequence)":
            excluded_from_feats.add(selected_entity_col)

        default_numeric_feats = [c for c in numeric_columns if c not in excluded_from_feats]

        selected_feature_cols = st.multiselect(
            "Numerical Sensor Feature Columns (F):",
            options=numeric_columns,
            default=default_numeric_feats,
            help="Select the continuous sensor signals used for model training and inference.",
        )

        # Label values configuration
        if selected_label_col != "None (Unsupervised / No Ground Truth)":
            c_lbl1, c_lbl2 = st.columns(2)
            with c_lbl1:
                normal_val_str = st.text_input("Normal Label Value", value="0")
            with c_lbl2:
                anom_val_str = st.text_input("Anomaly Label Value", value="1")
        else:
            normal_val_str = "0"
            anom_val_str = "1"

    # Display Computed Matrix Dimensions
    T_dim = (
        sum(series.n_timesteps for series in benchmark_series_list)
        if benchmark_series_list is not None else len(loaded_df)
    )
    F_dim = len(selected_feature_cols)

    dim_col1, dim_col2 = st.columns([2, 1])
    with dim_col1:
        st.markdown(
            f"""
            <div style="background: #181926; border: 1px solid #38BDF8; border-radius: 8px; padding: 14px; margin-top: 10px;">
                <span style="color: #38BDF8; font-weight: 700; font-size: 15px;">Target Feature Matrix Shape:</span>
                <span style="color: #F8FAFC; font-weight: 600; font-size: 16px; margin-left: 8px;">
                    X.shape = ({T_dim:,}, {F_dim})
                </span>
                <span style="color: #94A3B8; font-size: 13px; margin-left: 14px;">
                    ({T_dim:,} timesteps × {F_dim} channels)
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with dim_col2:
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        confirm_btn = st.button("Confirm Selection & Save", type="primary", use_container_width=True)

    if confirm_btn:
        if not selected_feature_cols:
            st.error("Please select at least one numeric feature column.")
            return

        # New input invalidates downstream artifacts from the previous run.
        for key in (
            "preprocessing_complete", "teacher_complete", "teacher_probs_train",
            "teacher_probs_val", "teacher_train_indices", "teacher_val_indices",
            "split_manifest", "split_stats", "split_description",
            "trained_model", "training_complete", "training_result", "last_checkpoint_path",
            "selected_threshold", "selected_window_threshold", "test_metrics",
            "test_step_metrics", "eval_complete", "inference_complete",
        ):
            st.session_state.pop(key, None)

        # Prepare and save into session_state
        st.session_state["raw_df"] = loaded_df
        st.session_state["data_filename"] = source_filename
        st.session_state["selected_features"] = selected_feature_cols
        st.session_state["time_col"] = selected_time_col if selected_time_col != "Use row index" else None
        st.session_state["label_col"] = (
            selected_label_col if selected_label_col != "None (Unsupervised / No Ground Truth)" else None
        )
        st.session_state["normal_label_val"] = normal_val_str
        st.session_state["anomaly_label_val"] = anom_val_str
        st.session_state["entity_col"] = (
            selected_entity_col if selected_entity_col != "None (Single Continuous Sequence)" else None
        )
        st.session_state["is_synthetic"] = is_synthetic
        st.session_state["data_loaded"] = True
        st.session_state["has_ground_truth"] = (
            selected_label_col != "None (Unsupervised / No Ground Truth)" or is_synthetic
        )
        st.session_state["benchmark_series_list"] = benchmark_series_list
        st.session_state["benchmark_key"] = benchmark_key
        st.session_state["benchmark_config"] = dict(selected_cfg) if benchmark_key else {}
        st.session_state["app_config"] = app_config
        st.session_state["benchmark_split_by"] = (
            "event" if benchmark_key == "metropt" else
            "subject" if benchmark_key == "daphnet" else
            "instance" if benchmark_key == "oil3w" else
            "chronological"
        )

        st.success("Configuration successfully saved to session state. You may proceed to Step 3 (Preprocessing).")

    # Feature Distribution Histograms (Plotly)
    if selected_feature_cols:
        st.markdown("---")
        st.subheader("Feature Distributions")
        n_plots = min(6, len(selected_feature_cols))
        feats_to_plot = selected_feature_cols[:n_plots]

        cols_per_row = 3
        n_rows = (n_plots + cols_per_row - 1) // cols_per_row
        fig = make_subplots(
            rows=n_rows,
            cols=cols_per_row,
            subplot_titles=[f"<b>{col}</b>" for col in feats_to_plot],
            vertical_spacing=0.15,
        )

        colors = ["#38BDF8", "#818CF8", "#A78BFA", "#F472B6", "#34D399", "#FBBF24"]

        for idx, col_name in enumerate(feats_to_plot):
            r = (idx // cols_per_row) + 1
            c = (idx % cols_per_row) + 1
            col_vals = loaded_df[col_name].dropna().values
            fig.add_trace(
                go.Histogram(
                    x=col_vals,
                    name=col_name,
                    marker_color=colors[idx % len(colors)],
                    opacity=0.75,
                    showlegend=False,
                ),
                row=r,
                col=c,
            )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#181926",
            plot_bgcolor="#13141f",
            font={"family": "Inter, sans-serif", "color": "#E2E8F0"},
            height=280 * n_rows,
            margin={"l": 40, "r": 20, "t": 40, "b": 40},
        )
        fig.update_xaxes(showgrid=True, gridcolor="#2e3148")
        fig.update_yaxes(showgrid=True, gridcolor="#2e3148")

        st.plotly_chart(fig, use_container_width=True)
