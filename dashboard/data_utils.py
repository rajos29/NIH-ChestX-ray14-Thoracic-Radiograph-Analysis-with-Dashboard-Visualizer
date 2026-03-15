# parquet loading + patient/image lookup + row prep

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


SELECTED_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Effusion",
    "Emphysema",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumothorax",
]


DISPLAY_LABELS = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Consolidation": "Consolidation",
    "Effusion": "Effusion",
    "Emphysema": "Emphysema",
    "Infiltration": "Infiltration",
    "Mass": "Mass",
    "Nodule": "Nodule",
    "Pleural_Thickening": "Pleural Thickening",
    "Pneumothorax": "Pneumothorax",
}

def load_manifest(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path).copy()

    sort_cols = [c for c in ["patient_id", "follow_up", "image_index"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


def build_indexes(df: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str], List[str]]:
    patient_to_rows = {
        str(pid): group.reset_index(drop=True)
        for pid, group in df.groupby("patient_id", sort=False)
    }

    image_to_patient = {
        str(img): str(pid)
        for img, pid in zip(df["image_index"], df["patient_id"])
    }

    patient_ids = sorted(patient_to_rows.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x))
    return patient_to_rows, image_to_patient, patient_ids


def get_patient_df(
    patient_to_rows: Dict[str, pd.DataFrame],
    patient_id: str,
) -> Optional[pd.DataFrame]:
    return patient_to_rows.get(str(patient_id))


def resolve_patient_from_image(
    image_to_patient: Dict[str, str],
    image_index: str,
) -> Optional[str]:
    return image_to_patient.get(str(image_index))


def find_image_position(patient_df: pd.DataFrame, image_index: str) -> int:
    matches = patient_df.index[patient_df["image_index"].astype(str) == str(image_index)].tolist()
    return matches[0] if matches else 0


def safe_str(value) -> str:
    if pd.isna(value):
        return "N/A"
    return str(value)


def make_case_summary(row: pd.Series) -> dict:
    return {
        "Patient ID": safe_str(row.get("patient_id")),
        "Image ID": safe_str(row.get("image_index")),
        "Split": safe_str(row.get("split")),
        "Follow-up": safe_str(row.get("follow_up")),
        "Age": safe_str(row.get("patient_age_years", row.get("patient_age"))),
        "Gender": safe_str(row.get("patient_gender")),
        "View Position": safe_str(row.get("view_position")),
        "Finding Labels": safe_str(row.get("finding_labels")),
        "Image Path": safe_str(row.get("image_path")),
    }


def build_prediction_table(row: pd.Series) -> pd.DataFrame:
    records = []

    for label in SELECTED_LABELS:
        gt_col = f"{label}_true"
        r_prob_col = f"{label}_resnet_prob"
        r_pred_col = f"{label}_resnet_pred"
        d_prob_col = f"{label}_densenet_prob"
        d_pred_col = f"{label}_densenet_pred"

        gt = row.get(gt_col, None)
        r_prob = row.get(r_prob_col, None)
        r_pred = row.get(r_pred_col, None)
        d_prob = row.get(d_prob_col, None)
        d_pred = row.get(d_pred_col, None)

        records.append(
            {
                "Disease": DISPLAY_LABELS[label],
                "Ground Truth": _fmt_binary(gt),
                "ResNet18": _fmt_model_cell(r_prob, r_pred),
                "DenseNet121": _fmt_model_cell(d_prob, d_pred),
            }
        )

    return pd.DataFrame(records)


def _fmt_binary(x) -> str:
    if pd.isna(x):
        return "N/A"
    return "Positive" if int(x) == 1 else "Negative"


def _fmt_model_cell(prob, pred) -> str:
    if pd.isna(prob) or pd.isna(pred):
        return "N/A"
    pred_text = "Positive" if int(pred) == 1 else "Negative"
    return f"{float(prob):.3f} · {pred_text}"


def get_image_path(row: pd.Series) -> Optional[str]:
    image_path = row.get("image_path", None)
    if image_path is None or pd.isna(image_path):
        return None
    return str(image_path)

def exists_image(path_str: Optional[str]) -> bool:
    if not path_str:
        return False
    return Path(path_str).exists()

def count_positive_truths(row: pd.Series) -> int:
    count = 0
    for label in SELECTED_LABELS:
        val = row.get(f"{label}_true", None)
        if pd.notna(val) and int(val) == 1:
            count += 1
    return count

def make_severity_text(row: pd.Series) -> str:
    positive_count = count_positive_truths(row)

    if positive_count == 0:
        return "No positive ground-truth findings"
    if positive_count == 1:
        return "Single positive finding present"
    if positive_count <= 3:
        return "Multiple positive findings present"
    return "High label burden in current image"

