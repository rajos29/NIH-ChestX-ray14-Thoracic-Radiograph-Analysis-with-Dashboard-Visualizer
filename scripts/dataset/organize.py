import pandas as pd

# Load CSV
df = pd.read_csv("sample/sample_labels.csv")

# Save as Parquet
df.to_parquet("manifest.parquet", engine="pyarrow", index=False)
