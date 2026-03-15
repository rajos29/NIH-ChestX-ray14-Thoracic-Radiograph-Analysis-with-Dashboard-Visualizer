import pandas as pd

MANIFEST = "manifest/manifest_v2.parquet"
MIN_TEST_POSITIVES = 20   # adjust if needed

df = pd.read_parquet(MANIFEST)

# Columns that are NOT disease labels
non_label_cols = {
    "image_index", "finding_labels", "follow_up", "patient_id",
    "patient_age", "patient_gender", "view_position",
    "orig_width", "orig_height", "pixel_spacing_x", "pixel_spacing_y",
    "patient_age_years", "image_path", "split"
}

# Detect binary label columns automatically
label_cols = [
    c for c in df.columns
    if c not in non_label_cols
    and set(df[c].dropna().unique()).issubset({0, 1})
]

# Count positives in test
test_counts = df[df["split"] == "test"][label_cols].sum()

# Select labels meeting threshold
selected_labels = test_counts[test_counts >= MIN_TEST_POSITIVES].index.tolist()

print("Selected labels (>= {} test positives):".format(MIN_TEST_POSITIVES))
print(selected_labels)

print(df[df["split"]=="train"][selected_labels].mean().sort_values(ascending=False))


