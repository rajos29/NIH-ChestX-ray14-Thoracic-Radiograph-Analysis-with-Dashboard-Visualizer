import pandas as pd
df = pd.read_parquet("manifest_v2.parquet")
print(df.groupby("split")["patient_id"].nunique())
