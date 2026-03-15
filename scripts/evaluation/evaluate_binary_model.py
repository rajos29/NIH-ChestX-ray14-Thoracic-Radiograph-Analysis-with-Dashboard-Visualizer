import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score


MANIFEST = "manifest_v2.parquet"
CKPT = "best_resnet18_binary.pt"
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class CXRBinaryDataset(Dataset):
    def __init__(self, df, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        x = self.transform(img)
        y = 1.0 if int(row["No Finding"]) == 0 else 0.0
        return x, torch.tensor([y], dtype=torch.float32)


def threshold_for_specificity(probs, targets, target_spec=0.90):
    neg_probs = probs[targets == 0]
    thr = np.quantile(neg_probs, target_spec)
    return float(thr)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    probs, targets = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        logits = model(x)
        p = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        probs.append(p)
        targets.append(y.numpy().reshape(-1))
    return np.concatenate(probs), np.concatenate(targets)


def main():
    df = pd.read_parquet(MANIFEST)

    df_val = df[df["split"] == "val"]
    df_test = df[df["split"] == "test"]

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    val_loader = DataLoader(CXRBinaryDataset(df_val, tfm),
                            batch_size=BATCH_SIZE, shuffle=False)

    test_loader = DataLoader(CXRBinaryDataset(df_test, tfm),
                             batch_size=BATCH_SIZE, shuffle=False)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model.to(DEVICE)

    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    val_probs, val_targets = predict(model, val_loader)
    test_probs, test_targets = predict(model, test_loader)

    # AUC
    print("Val AUC:", roc_auc_score(val_targets, val_probs))
    print("Test AUC:", roc_auc_score(test_targets, test_probs))

    # Threshold at 90% specificity on val
    thr = threshold_for_specificity(val_probs, val_targets, 0.90)
    print("Threshold (90% specificity):", thr)

    test_pred = (test_probs >= thr).astype(int)

    print("\nConfusion Matrix (Test):")
    print(confusion_matrix(test_targets, test_pred))

    print("\nClassification Report (Test):")
    print(classification_report(test_targets, test_pred, digits=4))


if __name__ == "__main__":
    main()
