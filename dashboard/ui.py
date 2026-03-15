from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #0A1929;
            color: #E0E0E0;
        }

        .block-container {
            padding-top: 4rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
            max-width: 1400px;
        }

        .app-title {
            font-size: 2.1rem;
            font-weight: 700;
            color: #E0E0E0;
            margin-bottom: 0.25rem;
            letter-spacing: -0.02em;
        }

        .app-subtitle {
            font-size: 1rem;
            color: #9CA3AF;
            margin-bottom: 1.2rem;
        }

        .card {
            background-color: #121212;
            border: 1px solid #1F2937;
            border-radius: 18px;
            padding: 1.1rem;
            margin-bottom: 1rem;
            box-shadow: 0 6px 18px rgba(0,0,0,0.32);
        }

        .card-title {
            font-size: 1rem;
            font-weight: 700;
            color: #66B2FF;
            margin-bottom: 0.9rem;
        }

        .meta-key {
            color: #9CA3AF;
            font-size: 0.84rem;
            margin-bottom: 0.15rem;
        }

        .meta-value {
            color: #E0E0E0;
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
        }

        .badge {
            display: inline-block;
            padding: 0.34rem 0.78rem;
            border-radius: 999px;
            border: 1px solid #334155;
            background-color: #0F172A;
            color: #E0E0E0;
            font-size: 0.85rem;
            margin-right: 0.45rem;
            margin-bottom: 0.45rem;
        }

        .badge-positive {
            background-color: rgba(153,255,204,0.10);
            color: #99FFCC;
            border: 1px solid rgba(153,255,204,0.35);
        }

        .badge-warning {
            background-color: rgba(255,204,102,0.10);
            color: #FFCC66;
            border: 1px solid rgba(255,204,102,0.35);
        }

        .badge-neutral {
            background-color: rgba(102,178,255,0.10);
            color: #66B2FF;
            border: 1px solid rgba(102,178,255,0.35);
        }

        .image-frame {
            background: linear-gradient(180deg, #050B14 0%, #0A0F18 100%);
            border: 1px solid #1F2937;
            border-radius: 18px;
            padding: 0.9rem;
        }

        .image-caption {
            color: #9CA3AF;
            font-size: 0.82rem;
            margin-top: 0.6rem;
        }

        .severity-box {
            background: linear-gradient(135deg, rgba(102,178,255,0.12), rgba(153,255,204,0.06));
            border: 1px solid #1F2937;
            border-radius: 16px;
            padding: 0.85rem 1rem;
            margin-bottom: 1rem;
        }

        .severity-title {
            color: #9CA3AF;
            font-size: 0.84rem;
            margin-bottom: 0.3rem;
        }

        .severity-value {
            color: #E0E0E0;
            font-size: 1.1rem;
            font-weight: 700;
        }

        .prediction-row {
            background-color: #0F172A;
            border: 1px solid #1F2937;
            border-radius: 14px;
            padding: 0.85rem 0.95rem;
            margin-bottom: 0.7rem;
        }

        .prediction-label {
            color: #E0E0E0;
            font-size: 0.96rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .prediction-meta {
            color: #9CA3AF;
            font-size: 0.82rem;
            margin-bottom: 0.25rem;
        }

        .chip {
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
        }

        .chip-positive {
            background-color: rgba(153,255,204,0.12);
            color: #99FFCC;
            border: 1px solid rgba(153,255,204,0.35);
        }

        .chip-negative {
            background-color: rgba(224,224,224,0.08);
            color: #E0E0E0;
            border: 1px solid rgba(224,224,224,0.18);
        }

        .chip-warning {
            background-color: rgba(255,204,102,0.12);
            color: #FFCC66;
            border: 1px solid rgba(255,204,102,0.35);
        }

        .compare-grid {
            display: grid;
            grid-template-columns: 1.1fr 0.8fr 1fr 1fr;
            gap: 0.75rem;
            align-items: start;
        }

        .compare-grid-header {
            align-items: center;
        }

        .compare-header {
            color: #66B2FF;
            font-size: 0.82rem;
            font-weight: 700;
            margin-bottom: 0.3rem;
        }

        .split-note {
            color: #9CA3AF;
            font-size: 0.82rem;
            margin-top: 0.6rem;
        }

        section[data-testid="stSidebar"] {
            background-color: #121212;
            border-right: 1px solid #1F2937;
        }

        /* ---- BUTTONS ---- */

        .stButton > button {
            background-color: #66B2FF;
            color: #0A1929;
            border-radius: 10px;
            border: none;
            padding: 0.42rem 0.85rem;
            font-weight: 700;
            transition: all 0.2s ease;
        }

        .stButton > button:hover {
            background-color: #4FA3FF;
        }

        .stButton > button:disabled {
            background-color: #1F2937 !important;
            color: #6B7280 !important;
            border: 1px solid #374151 !important;
            cursor: not-allowed !important;
            opacity: 1 !important;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid #1F2937;
            border-radius: 14px;
        }

        /* ---- INPUTS / SELECTS ---- */

        section[data-testid="stSidebar"] .stTextInput input {
            background-color: #0F172A !important;
            color: #E0E0E0 !important;
            border: 1px solid #1F2937 !important;
            border-radius: 10px !important;
        }

        section[data-testid="stSidebar"] .stTextInput input::placeholder {
            color: #9CA3AF !important;
        }

        section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
            background-color: #0F172A !important;
            color: #E0E0E0 !important;
            border: 1px solid #1F2937 !important;
            border-radius: 10px !important;
        }

        section[data-testid="stSidebar"] [data-baseweb="select"] svg {
            fill: #E0E0E0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown('<div class="app-title">Chest X-Ray Case Review Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="app-subtitle">Interactive case review for chest radiographs, metadata, and multi-model predictions.</div>',
        unsafe_allow_html=True,
    )


def render_case_badges(summary: dict) -> None:
    badges = [
        f"Patient {summary.get('Patient ID', 'N/A')}",
        f"Image {summary.get('Image ID', 'N/A')}",
        f"Split: {summary.get('Split', 'N/A')}",
        f"View: {summary.get('View Position', 'N/A')}",
    ]
    badge_html = "".join([f'<span class="badge badge-neutral">{b}</span>' for b in badges])
    st.markdown(badge_html, unsafe_allow_html=True)


def render_severity_box(severity_text: str, positive_count: int) -> None:
    st.markdown(
        f"""
        <div class="severity-box">
            <div class="severity-title">Case Summary</div>
            <div class="severity-value">{severity_text}</div>
            <div class="split-note">Ground-truth positive labels in this image: {positive_count}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metadata_card(summary: dict) -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Case Metadata</div>', unsafe_allow_html=True)

    fields = [
        "Patient ID",
        "Image ID",
        "Split",
        "Follow-up",
        "Age",
        "Gender",
        "View Position",
        "Finding Labels",
    ]

    for field in fields:
        st.markdown(f'<div class="meta-key">{field}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="meta-value">{summary.get(field, "N/A")}</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def render_image_panel(image_source, image_mode: str = "Original X-ray") -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f'<div class="card-title">{image_mode}</div>', unsafe_allow_html=True)
    st.markdown('<div class="image-frame">', unsafe_allow_html=True)

    if image_source is None:
        st.warning("Image could not be displayed.")
    elif isinstance(image_source, str):
        if Path(image_source).exists():
            st.image(image_source, use_container_width=True)
            st.markdown(
                '<div class="image-caption">Displayed in radiograph-first review mode.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.warning("Image file could not be found.")
    else:
        st.image(image_source, use_container_width=True)
        st.markdown(
            '<div class="image-caption">Grad-CAM overlay highlighting image regions contributing most strongly to the selected class response.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)



def render_prediction_table(pred_df: pd.DataFrame) -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Ground Truth vs Model Predictions</div>', unsafe_allow_html=True)
    st.dataframe(pred_df, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_prediction_cards(pred_df: pd.DataFrame) -> None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">Prediction Review</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="compare-grid compare-grid-header" style="margin-bottom:0.75rem;">
            <div class="compare-header">Disease</div>
            <div class="compare-header">Ground Truth</div>
            <div class="compare-header">ResNet18</div>
            <div class="compare-header">DenseNet121</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for _, row in pred_df.iterrows():
        disease = row["Disease"]
        gt = row["Ground Truth"]
        resnet = row["ResNet18"]
        densenet = row["DenseNet121"]

        st.markdown(
            f"""
            <div class="prediction-row">
                <div class="compare-grid">
                    <div class="prediction-label" style="margin-bottom:0;">{disease}</div>
                    <div>{_truth_badge_block(gt)}</div>
                    <div>{_render_model_cell_html(resnet)}</div>
                    <div>{_render_model_cell_html(densenet)}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="split-note">Train predictions can be reviewed visually, but held-out validation/test images remain the appropriate basis for model performance claims.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_patient_navigation_info(current_idx: int, total_patients: int, image_idx: int, total_images: int) -> None:
    col1, col2 = st.columns(2)
    with col1:
        st.caption(f"Patient {current_idx + 1} of {total_patients}")
    with col2:
        st.caption(f"Image {image_idx + 1} of {total_images}")


def _truth_badge_block(gt: str) -> str:
    if gt == "Positive":
        return '<span class="chip chip-positive">Positive</span>'
    if gt == "Negative":
        return '<span class="chip chip-negative">Negative</span>'
    return '<span class="chip chip-warning">Unknown</span>'


def _render_model_cell_html(value: str) -> str:
    if value == "N/A":
        return '<span class="chip chip-warning">N/A</span>'

    prob_text, pred_text = [x.strip() for x in value.split("·")]
    chip_class = "chip-positive" if pred_text == "Positive" else "chip-negative"

    try:
        prob = float(prob_text)
        pct = int(round(prob * 100))
    except Exception:
        pct = 0

    bar_html = f"""
    <div class="prediction-meta">Probability: {prob_text}</div>
    <div style="background:#1F2937; border-radius:999px; height:10px; overflow:hidden; margin:0.25rem 0 0.45rem 0;">
        <div style="width:{pct}%; height:10px; background:linear-gradient(90deg, #66B2FF 0%, #99FFCC 100%);"></div>
    </div>
    <span class="chip {chip_class}">{pred_text}</span>
    """
    return bar_html
