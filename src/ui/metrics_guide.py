"""Page: the metrics guide (METRICS_GUIDE.md)."""

from __future__ import annotations

import streamlit as st

from src.app import core


def show() -> None:
    path = core.ROOT / "METRICS_GUIDE.md"
    st.markdown(path.read_text(encoding="utf-8") if path.exists() else "METRICS_GUIDE.md not found.")
