import pandas as pd
df = pd.read_parquet("manifest_v2.parquet")
print((df.groupby("patient_id")["split"].nunique() > 1).sum())
