from __future__ import annotations

from pathlib import Path

import streamlit as st


STYLE_PATHS = (
    Path(__file__).resolve().parent / "assets" / "base.css",
    Path(__file__).resolve().parent / "assets" / "components.css",
)


def inject_global_style() -> None:
    css = "\n".join(path.read_text(encoding="utf-8") for path in STYLE_PATHS)
    st.markdown(f"<style>\n{css}\n</style>", unsafe_allow_html=True)
