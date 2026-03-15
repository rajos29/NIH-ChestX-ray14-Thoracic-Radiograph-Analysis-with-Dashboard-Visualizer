import pandas as pd

df = pd.read_parquet("manifest_v2.parquet")

# Non-label columns (everything that is NOT a disease)
non_label_cols = {
    "image_index", "finding_labels", "follow_up", "patient_id",
    "patient_age", "patient_gender", "view_position",
    "orig_width", "orig_height", "pixel_spacing_x", "pixel_spacing_y",
    "patient_age_years", "image_path", "split"
}

# Automatically detect disease columns (binary 0/1)
label_cols = []
for c in df.columns:
    if c not in non_label_cols:
        if set(df[c].dropna().unique()).issubset({0, 1}):
            label_cols.append(c)

print("Checking disease presence across splits...\n")

for label in label_cols:
    presence = df.groupby("split")[label].sum()
    print(f"{label}:")
    print(presence)
    print()
