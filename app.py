"""Streamlit app for the IE 643 project: VLM-guided knowledge distillation for
time-series anomaly detection (final system).

    .venv\\Scripts\\python.exe -m streamlit run app.py
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "cache" / "huggingface"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.app import core
from src.ui import data_pipeline, evaluation, inference, metrics_guide, overview, results, student, teacher

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

st.set_page_config(page_title="VLM-KD Anomaly Detection", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")

PAGES = {
    "1 · Overview": overview,
    "2 · Data pipeline": data_pipeline,
    "3 · Teacher (frozen CLIP)": teacher,
    "4 · Student & training": student,
    "5 · Evaluation (metrics step by step)": evaluation,
    "6 · Teacher vs student on a CSV": inference,
    "7 · Results & checks": results,
    "8 · Metrics guide": metrics_guide,
}

with st.sidebar:
    st.markdown("### VLM-KD anomaly detection\nIE 643 course project")
    st.selectbox("Dataset", core.DATASETS, format_func=lambda n: core.PRETTY[n], key="dataset",
                 help="Every page shows this dataset's data, models and results.")
    page = st.radio("Page", list(PAGES), key="page")
    st.caption("Teacher: CLIP ViT-B/32, frozen, learned prompts on heatmap + spectrogram views.\n\n"
               "Student: Conv1d → BiGRU, ≈100k parameters, distilled from the teacher.")
    live = st.session_state.get("live_models", {})
    if live:
        st.caption(f"Live-trained models this session: {len(live)}")

try:
    PAGES[page].show()
except Exception as exc:  # show the error on the page instead of a blank screen
    st.error(f"Error on page '{page}': {exc}")
    logging.getLogger("app").exception("page error")
