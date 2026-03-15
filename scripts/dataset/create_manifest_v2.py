import re
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ================== CONFIG ==================
INPUT_PARQUET = "manifest.parquet"      # your current manifest1 parquet
IMAGES_DIR = "images"                      # folder with png files
OUTPUT_PARQUET = "manifest_v2.parquet"     # output model-ready manifest
RANDOM_SEED = 42
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15
# ===========================================

def parse_age_to_years(x):
    """Convert '060Y' -> 60. Returns pandas.NA if unknown/bad."""
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()
    m = re.match(r"^(\d+)\s*[Yy]$", s)
    if m:
        return int(m.group(1))
    # already numeric?
    try:
        return int(float(s))
    except Exception:
        return pd.NA

def split_labels(s):
    """Split 'A|B|C' -> ['A','B','C']"""
    if pd.isna(s):
        return []
    s = str(s).strip()
    if not s:
        return []
    return [x.strip() for x in s.split("|") if x.strip()]

def main():
    # Load
    df = pd.read_parquet(INPUT_PARQUET)

    # Standardize column names (keeps your original data, just renames headers)
    rename_map = {
        "Image Index": "image_index",
        "Finding Labels": "finding_labels",
        "Follow-up #": "follow_up",
        "Patient ID": "patient_id",
        "Patient Age": "patient_age",
        "Patient Gender": "patient_gender",
        "View Position": "view_position",
        "OriginalImageWidth": "orig_width",
        "OriginalImageHeight": "orig_height",
        "OriginalImagePixelSpacing_x": "pixel_spacing_x",
        "OriginalImagePixelSpacing_y": "pixel_spacing_y",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Required columns
    required = ["image_index", "finding_labels", "patient_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Types
    df["patient_id"] = df["patient_id"].astype(str)

    # Age numeric
    if "patient_age" in df.columns:
        df["patient_age_years"] = df["patient_age"].apply(parse_age_to_years)

    # Image path column for your dataloader
    df["image_path"] = df["image_index"].apply(lambda x: str(Path(IMAGES_DIR) / str(x)))

    # Build multi-hot label columns
    label_lists = df["finding_labels"].apply(split_labels)
    all_labels = sorted({lab for labs in label_lists for lab in labs})

    for lab in all_labels:
        df[lab] = label_lists.apply(lambda labs, lab=lab: int(lab in labs))

    # Sanity: "No Finding" should not co-occur with other labels
    if "No Finding" in df.columns:
        other = [c for c in all_labels if c != "No Finding"]
        if other:
            conflicts = df[(df["No Finding"] == 1) & (df[other].sum(axis=1) > 0)]
            if len(conflicts) > 0:
                print(f"[WARN] {len(conflicts)} rows where 'No Finding' co-occurs with other labels.")

    # Patient-level split (70/15/15 by default)
    assert abs((TRAIN_FRAC + VAL_FRAC + TEST_FRAC) - 1.0) < 1e-9

    unique_patients = df["patient_id"].unique()
    train_p, temp_p = train_test_split(
        unique_patients,
        test_size=(1.0 - TRAIN_FRAC),
        random_state=RANDOM_SEED,
        shuffle=True
    )

    # split temp into val/test in the right ratio
    val_ratio_of_temp = VAL_FRAC / (VAL_FRAC + TEST_FRAC)
    val_p, test_p = train_test_split(
        temp_p,
        test_size=(1.0 - val_ratio_of_temp),
        random_state=RANDOM_SEED,
        shuffle=True
    )

    train_set, val_set, test_set = set(train_p), set(val_p), set(test_p)

    def assign_split(pid: str) -> str:
        if pid in train_set:
            return "train"
        if pid in val_set:
            return "val"
        return "test"

    df["split"] = df["patient_id"].apply(assign_split)

    # Leakage check: must be 0
    leakage = (df.groupby("patient_id")["split"].nunique() > 1).sum()
    if int(leakage) != 0:
        raise RuntimeError(f"Patient leakage detected: {int(leakage)} patients in multiple splits.")

    # Quick report
    print("Rows:", len(df))
    print("Patients:", df["patient_id"].nunique())
    print("Split counts:\n", df["split"].value_counts())
    print("Num label columns:", len(all_labels))
    print("Example labels:", all_labels[:10])

    # Save
    df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"Saved -> {OUTPUT_PARQUET}")

if __name__ == "__main__":
    main()
