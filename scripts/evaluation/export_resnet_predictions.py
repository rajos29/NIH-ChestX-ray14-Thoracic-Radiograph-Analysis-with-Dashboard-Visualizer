import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

# ---- CONFIG ----
MANIFEST = "manifest/manifest_v2.parquet"
CKPT = "models/best_resnet18_multilabel.pt"
OUT_CSV = "manifest/resnet_predictions.csv"

BATCH_SIZE = 32
NUM_WORKERS = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
TARGET_SPLIT = "test"
TARGET_SPEC = 0.90
# ----------------

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

class CXRMultiLabelDataset(Dataset):
    def __init__(self, df, label_cols, transform):
        self.df = df.reset_index(drop=True)
        self.label_cols = label_cols
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        x = self.transform(img)
        y = torch.tensor(row[self.label_cols].values.astype(np.float32))
        image_index = row["image_index"]
        return x, y, image_index

def make_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

@torch.no_grad()
def predict_probs(model, loader):
    model.eval()
    probs_list, y_list, image_ids = [], [], []

    for x, y, image_index in loader:
        x = x.to(DEVICE, non_blocking=True)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()

        probs_list.append(probs)
        y_list.append(y.numpy())
        image_ids.extend(list(image_index))

    return (
        np.concatenate(probs_list, axis=0),
        np.concatenate(y_list, axis=0),
        image_ids,
    )

def threshold_for_specificity(probs, y_true, target_spec):
    neg_probs = probs[y_true == 0]
    if len(neg_probs) == 0:
        return 1.0
    return float(np.quantile(neg_probs, target_spec))

def main():
    os.makedirs("manifest", exist_ok=True)

    df = pd.read_parquet(MANIFEST)

    df_val = df[df["split"] == "val"].copy()
    df_target = df.copy()

    tfm = make_transform(IMG_SIZE)

    val_loader = DataLoader(
        CXRMultiLabelDataset(df_val, SELECTED_LABELS, tfm),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    target_loader = DataLoader(
        CXRMultiLabelDataset(df_target, SELECTED_LABELS, tfm),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(SELECTED_LABELS))
    model.to(DEVICE)

    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    val_probs, val_true, _ = predict_probs(model, val_loader)
    target_probs, target_true, target_image_ids = predict_probs(model, target_loader)

    thresholds = {}
    for i, label in enumerate(SELECTED_LABELS):
        thresholds[label] = threshold_for_specificity(
            val_probs[:, i],
            val_true[:, i].astype(int),
            TARGET_SPEC
        )

    out_df = pd.DataFrame({"image_index": target_image_ids})

    for i, label in enumerate(SELECTED_LABELS):
        probs = target_probs[:, i]
        preds = (probs >= thresholds[label]).astype(int)

        out_df[f"{label}_resnet_prob"] = probs
        out_df[f"{label}_resnet_pred"] = preds

    out_df.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}")
    print(f"Rows: {len(out_df)}")

if __name__ == "__main__":
    main()
