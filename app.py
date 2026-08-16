from __future__ import annotations

import streamlit as st

from dashboard.config import PRETRAIN_ROOT, RESULTS_DIR
from dashboard.data import load_physiology_summary, load_pretraining_data
from dashboard.services import build_pretraining_overview
from dashboard.views import render_app


st.set_page_config(page_title="fMRI/EEG产妇抑郁判断", layout="wide")


@st.cache_data
def _load_physiology():
    return load_physiology_summary(RESULTS_DIR)


@st.cache_data
def _load_pretraining():
    return load_pretraining_data(PRETRAIN_ROOT)


def main() -> None:
    summary = _load_physiology()
    pretraining = _load_pretraining()
    pretrain_overview = build_pretraining_overview(pretraining)
    render_app(summary, pretraining, pretrain_overview)


if __name__ == "__main__":
    main()
