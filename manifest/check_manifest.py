import pandas as pd

base_df = pd.read_parquet("manifest/manifest_v2.parquet")
resnet_df = pd.read_csv("manifest/resnet_predictions.csv")
densenet_df = pd.read_csv("manifest/densenet_predictions.csv")  # or your actual file name

print("BASE dtype:", base_df["image_index"].dtype)
print("RESNET dtype:", resnet_df["image_index"].dtype)
print("DENSENET dtype:", densenet_df["image_index"].dtype)

print("\nBASE sample:")
print(base_df["image_index"].head(10).tolist())

print("\nRESNET sample:")
print(resnet_df["image_index"].head(10).tolist())

print("\nDENSENET sample:")
print(densenet_df["image_index"].head(10).tolist())
import pandas as pd

df = pd.read_parquet("manifest/case_review_manifest.parquet")

print("Total rows:", len(df))

resnet_prob_cols = [c for c in df.columns if c.endswith("_resnet_prob")]
densenet_prob_cols = [c for c in df.columns if c.endswith("_densenet_prob")]

print("Rows with any ResNet prediction:",
      df[resnet_prob_cols].notna().any(axis=1).sum())

print("Rows with any DenseNet prediction:",
      df[densenet_prob_cols].notna().any(axis=1).sum())

print("\nBy split for ResNet:")
print(df.loc[df[resnet_prob_cols].notna().any(axis=1), "split"].value_counts(dropna=False))

print("\nBy split for DenseNet:")
print(df.loc[df[densenet_prob_cols].notna().any(axis=1), "split"].value_counts(dropna=False))
