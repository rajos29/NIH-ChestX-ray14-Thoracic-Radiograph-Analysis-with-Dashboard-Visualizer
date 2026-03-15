import os
import pandas as pd

# ---------------- CONFIG ----------------
BASE_MANIFEST = r"manifest/manifest_v2.parquet"
RESNET_PRED_CSV = r"manifest/resnet_predictions.csv"
DENSENET_PRED_CSV = r"manifest/densenet_predictions.csv"

OUT_DIR = r"manifest"
OUT_CSV = os.path.join(OUT_DIR, "case_review_manifest.csv")
OUT_PARQUET = os.path.join(OUT_DIR, "case_review_manifest.parquet")

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
# ----------------------------------------


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load base manifest
    df = pd.read_parquet(BASE_MANIFEST)

    # Required/desired metadata columns
    preferred_base_cols = [
        "image_index",
        "patient_id",
        "image_path",
        "follow_up",
        "patient_age",
        "patient_gender",
        "view_position",
        "finding_labels",
        "split",
        "orig_width",
        "orig_height",
        "pixel_spacing_x",
        "pixel_spacing_y",
        "patient_age_years",
    ]


    # Keep only columns that actually exist
    existing_base_cols = [c for c in preferred_base_cols if c in df.columns]

    # Keep truth label columns if present
    truth_cols_present = [c for c in SELECTED_LABELS if c in df.columns]

    if "image_index" not in df.columns:
        raise ValueError("Base manifest must contain 'image_index'.")

    if "patient_id" not in df.columns:
        print("WARNING: 'patient_id' not found in base manifest. Patient-based dashboard navigation will not work correctly.")

    # Build base dataframe
    base_df = df[existing_base_cols + truth_cols_present].copy()

    # Rename truth columns to *_true
    rename_truth = {label: f"{label}_true" for label in truth_cols_present}
    base_df = base_df.rename(columns=rename_truth)

    # Load model prediction CSVs
    resnet_df = pd.read_csv(RESNET_PRED_CSV)
    densenet_df = pd.read_csv(DENSENET_PRED_CSV)

    if "image_index" not in resnet_df.columns:
        raise ValueError("ResNet prediction CSV must contain 'image_index'.")
    if "image_index" not in densenet_df.columns:
        raise ValueError("DenseNet prediction CSV must contain 'image_index'.")

    # Merge predictions onto base image-level table
    merged = base_df.merge(resnet_df, on="image_index", how="left")
    merged = merged.merge(densenet_df, on="image_index", how="left")

    # Sort so each patient's images stay grouped together
    sort_cols = [c for c in ["patient_id", "follow_up", "image_index"] if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols).reset_index(drop=True)

    # Add per-patient image order for easier dashboard navigation
    if "patient_id" in merged.columns:
        merged["patient_image_order"] = merged.groupby("patient_id").cumcount() + 1
        merged["patient_image_count"] = merged.groupby("patient_id")["image_index"].transform("count")

    # Save outputs
    merged.to_csv(OUT_CSV, index=False)
    merged.to_parquet(OUT_PARQUET, index=False)

    # Basic sanity checks
    print(f"Saved CSV: {OUT_CSV}")
    print(f"Saved Parquet: {OUT_PARQUET}")
    print(f"Total rows: {len(merged)}")

    if "patient_id" in merged.columns:
        print(f"Unique patients: {merged['patient_id'].nunique()}")

    if "image_index" in merged.columns:
        print(f"Unique images: {merged['image_index'].nunique()}")

    # Check duplicates by image ID
    dup_count = merged["image_index"].duplicated().sum()
    print(f"Duplicate image_index rows: {dup_count}")

    # Show a sample of columns
    print("\nColumns in final manifest:")
    for c in merged.columns:
        print(f" - {c}")


if __name__ == "__main__":
    main()
