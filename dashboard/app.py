from __future__ import annotations

import streamlit as st

from data_utils import (
    build_indexes,
    build_prediction_table,
    count_positive_truths,
    find_image_position,
    get_image_path,
    get_patient_df,
    load_manifest,
    make_case_summary,
    make_severity_text,
    resolve_patient_from_image,
)
from gradcam_utils import (
    DISPLAY_LABELS,
    DISPLAY_TO_INTERNAL,
    SELECTED_LABELS,
    generate_gradcam_overlay,
    load_densenet_model,
    load_resnet_model,
)
from ui import (
    inject_css,
    render_case_badges,
    render_header,
    render_image_panel,
    render_metadata_card,
    render_patient_navigation_info,
    render_prediction_cards,
    render_prediction_table,
    render_severity_box,
)

MANIFEST_PATH = "manifest/case_review_manifest.parquet"
RESNET_CKPT = "models/best_resnet18_multilabel.pt"
DENSENET_CKPT = "models/best_densenet121_multilabel.pt"


@st.cache_data
def load_data():
    df = load_manifest(MANIFEST_PATH)
    patient_to_rows, image_to_patient, patient_ids = build_indexes(df)
    return df, patient_to_rows, image_to_patient, patient_ids


@st.cache_resource
def load_models():
    resnet_model = load_resnet_model(RESNET_CKPT)
    densenet_model = load_densenet_model(DENSENET_CKPT)
    return resnet_model, densenet_model


def init_session_state(patient_ids):
    if "patient_idx" not in st.session_state:
        st.session_state.patient_idx = 0
    if "image_idx" not in st.session_state:
        st.session_state.image_idx = 0
    if "selected_patient_id" not in st.session_state:
        st.session_state.selected_patient_id = patient_ids[0] if patient_ids else None


def set_patient(patient_ids, patient_id: str) -> None:
    if patient_id in patient_ids:
        st.session_state.selected_patient_id = patient_id
        st.session_state.patient_idx = patient_ids.index(patient_id)
        st.session_state.image_idx = 0


def clamp_image_idx(patient_df):
    if patient_df is None or len(patient_df) == 0:
        st.session_state.image_idx = 0
        return
    st.session_state.image_idx = max(0, min(st.session_state.image_idx, len(patient_df) - 1))


def main():
    st.set_page_config(
        page_title="Chest X-Ray Dashboard",
        page_icon="🩻",
        layout="wide",
    )

    inject_css()
    render_header()

    df, patient_to_rows, image_to_patient, patient_ids = load_data()
    resnet_model, densenet_model = load_models()

    if not patient_ids:
        st.error("No patients found in manifest.")
        return

    init_session_state(patient_ids)

    # ---------- SIDEBAR ----------
    st.sidebar.header("Navigation")

    patient_search = st.sidebar.text_input("Search by Patient ID")
    image_search = st.sidebar.text_input("Search by Image ID")

    patient_dropdown = st.sidebar.selectbox(
        "Or select Patient ID",
        options=patient_ids,
        index=st.session_state.patient_idx if patient_ids else 0,
    )

    image_mode = st.sidebar.radio(
        "Image View",
        ["Original X-ray", "Grad-CAM"],
        index=0,
    )

    gradcam_model_name = st.sidebar.selectbox(
        "Grad-CAM Model",
        options=["ResNet18", "DenseNet121"],
        index=0,
        disabled=(image_mode != "Grad-CAM"),
    )

    # Patient search
    if patient_search:
        patient_search = patient_search.strip()
        if patient_search in patient_ids and patient_search != st.session_state.selected_patient_id:
            set_patient(patient_ids, patient_search)
            st.rerun()

    # Image search
    if image_search:
        image_search = image_search.strip()
        resolved_patient = resolve_patient_from_image(image_to_patient, image_search)
        if resolved_patient is not None:
            set_patient(patient_ids, resolved_patient)
            patient_df_tmp = get_patient_df(patient_to_rows, resolved_patient)
            if patient_df_tmp is not None and not patient_df_tmp.empty:
                st.session_state.image_idx = find_image_position(patient_df_tmp, image_search)
            st.rerun()

    # Dropdown selection
    if patient_dropdown != st.session_state.selected_patient_id:
        set_patient(patient_ids, patient_dropdown)
        st.rerun()

    # ---------- CURRENT PATIENT ----------
    selected_patient_id = st.session_state.selected_patient_id
    patient_df = get_patient_df(patient_to_rows, selected_patient_id)

    if patient_df is None or patient_df.empty:
        st.error("Selected patient could not be loaded.")
        return

    clamp_image_idx(patient_df)

    # ---------- NAV BUTTONS ----------
    st.sidebar.markdown("---")

    at_first_patient = st.session_state.patient_idx <= 0
    at_last_patient = st.session_state.patient_idx >= len(patient_ids) - 1

    col1, col2 = st.sidebar.columns(2)
    prev_patient = col1.button("Prev Patient", disabled=at_first_patient)
    next_patient = col2.button("Next Patient", disabled=at_last_patient)

    st.sidebar.markdown("---")

    total_images = len(patient_df)
    at_first_image = st.session_state.image_idx <= 0
    at_last_image = total_images <= 1 or st.session_state.image_idx >= total_images - 1

    col3, col4 = st.sidebar.columns(2)
    prev_image = col3.button("Prev Image", disabled=at_first_image)
    next_image = col4.button("Next Image", disabled=at_last_image)

    if prev_patient:
        new_idx = max(0, st.session_state.patient_idx - 1)
        set_patient(patient_ids, patient_ids[new_idx])
        st.rerun()

    if next_patient:
        new_idx = min(len(patient_ids) - 1, st.session_state.patient_idx + 1)
        set_patient(patient_ids, patient_ids[new_idx])
        st.rerun()

    if prev_image:
        st.session_state.image_idx = max(0, st.session_state.image_idx - 1)
        st.rerun()

    if next_image:
        st.session_state.image_idx = min(len(patient_df) - 1, st.session_state.image_idx + 1)
        st.rerun()

    # ---------- MAIN CONTENT ----------
    clamp_image_idx(patient_df)
    image_idx = st.session_state.image_idx

    selected_image_row = patient_df.iloc[image_idx]
    summary = make_case_summary(selected_image_row)
    pred_df = build_prediction_table(selected_image_row)
    severity_text = make_severity_text(selected_image_row)
    positive_count = count_positive_truths(selected_image_row)

    render_case_badges(summary)

    image_options = patient_df["image_index"].astype(str).tolist()
    chosen_image = st.selectbox(
        "Select image within patient",
        options=image_options,
        index=image_idx,
        key="main_image_selectbox",
    )

    if chosen_image != str(selected_image_row["image_index"]):
        st.session_state.image_idx = image_options.index(chosen_image)
        st.rerun()

    # Recompute after image selectbox
    clamp_image_idx(patient_df)
    image_idx = st.session_state.image_idx
    selected_image_row = patient_df.iloc[image_idx]

    summary = make_case_summary(selected_image_row)
    pred_df = build_prediction_table(selected_image_row)
    severity_text = make_severity_text(selected_image_row)
    positive_count = count_positive_truths(selected_image_row)

    # Choose default Grad-CAM disease from strongest prediction of selected model
    if gradcam_model_name == "ResNet18":
        prob_cols = {DISPLAY_LABELS[label]: selected_image_row.get(f"{label}_resnet_prob", 0.0) for label in SELECTED_LABELS}
    else:
        prob_cols = {DISPLAY_LABELS[label]: selected_image_row.get(f"{label}_densenet_prob", 0.0) for label in SELECTED_LABELS}

    default_gradcam_label = max(prob_cols, key=lambda k: float(prob_cols[k]) if prob_cols[k] == prob_cols[k] else -1.0)

    gradcam_display_label = st.selectbox(
        "Grad-CAM Target Disease",
        options=list(DISPLAY_LABELS.values()),
        index=list(DISPLAY_LABELS.values()).index(default_gradcam_label),
        disabled=(image_mode != "Grad-CAM"),
    )

    render_patient_navigation_info(
        current_idx=st.session_state.patient_idx,
        total_patients=len(patient_ids),
        image_idx=st.session_state.image_idx,
        total_images=len(patient_df),
    )

    # Render image source
    image_source = get_image_path(selected_image_row)

    if image_mode == "Grad-CAM":
        try:
            target_label = DISPLAY_TO_INTERNAL[gradcam_display_label]
            if gradcam_model_name == "ResNet18":
                overlay_img, _ = generate_gradcam_overlay(
                    resnet_model,
                    "ResNet18",
                    image_source,
                    target_label,
                )
            else:
                overlay_img, _ = generate_gradcam_overlay(
                    densenet_model,
                    "DenseNet121",
                    image_source,
                    target_label,
                )
            image_source = overlay_img
        except Exception as e:
            st.warning(f"Grad-CAM generation failed: {e}")

    left, right = st.columns([1.35, 1.0], gap="large")

    with left:
        render_image_panel(image_source, image_mode=image_mode)

    with right:
        render_severity_box(severity_text, positive_count)
        render_metadata_card(summary)

    render_prediction_cards(pred_df)

    with st.expander("Show raw comparison table"):
        render_prediction_table(pred_df)


if __name__ == "__main__":
    main()
