# train_binary_baseline.py
# Binary baseline: abnormal vs normal (abnormal = No Finding == 0)
# Reads: manifest_v2.parquet (must include: image_path, split, No Finding)
# Trains: pretrained ResNet18 on ImageNet with 1-logit head
# Reports: ROC-AUC on val + test

import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import roc_auc_score


# ----------------- CONFIG -----------------
@dataclass
class CFG:
    manifest_path: str = "manifest_v2.parquet"
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 5
    lr: float = 3e-4
    weight_decay: float = 1e-4
    img_size: int = 224
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    freeze_backbone_epochs: int = 1  # freeze for first epoch to stabilize on small data
    amp: bool = True  # mixed precision on CUDA
# -----------------------------------------


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # speed; set False for strict determinism


class CXRBinaryDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row["image_path"]

        # Load image (NIH images are typically grayscale; convert to 3ch for ResNet)
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Failed to load image at '{path}': {e}")

        x = self.transform(img)

        # abnormal = (No Finding == 0)
        # Expect No Finding column exists and is 0/1
        y = 1.0 if int(row["No Finding"]) == 0 else 0.0
        y = torch.tensor([y], dtype=torch.float32)  # shape (1,)

        return x, y


def make_transforms(img_size: int):
    # ImageNet normalization (since backbone is ImageNet-pretrained)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=7),
        transforms.ToTensor(),
        normalize,
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])

    return train_tf, eval_tf


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    probs = []
    targets = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        p = torch.sigmoid(logits)

        probs.append(p.detach().cpu().numpy().reshape(-1))
        targets.append(y.detach().cpu().numpy().reshape(-1))

    probs = np.concatenate(probs, axis=0)
    targets = np.concatenate(targets, axis=0)
    return probs, targets


def compute_auc(probs, targets):
    # AUC requires both classes present
    if len(np.unique(targets)) < 2:
        return float("nan")
    return roc_auc_score(targets, probs)


def main():
    cfg = CFG()
    set_seed(cfg.seed)
    print("Device:", cfg.device)

    # Load manifest
    df = pd.read_parquet(cfg.manifest_path)

    required_cols = {"image_path", "split", "No Finding"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {missing}")

    # Basic sanity
    for split in ["train", "val", "test"]:
        if split not in set(df["split"].unique()):
            raise ValueError(f"Split '{split}' not found in manifest. Found: {df['split'].unique()}")

    # Split dataframes
    df_train = df[df["split"] == "train"].copy()
    df_val = df[df["split"] == "val"].copy()
    df_test = df[df["split"] == "test"].copy()

    # (Optional) quick file existence check on a small sample
    for p in df_train["image_path"].head(5).tolist():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Image path does not exist: {p}")

    train_tf, eval_tf = make_transforms(cfg.img_size)

    ds_train = CXRBinaryDataset(df_train, train_tf)
    ds_val = CXRBinaryDataset(df_val, eval_tf)
    ds_test = CXRBinaryDataset(df_test, eval_tf)

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False,
                         num_workers=cfg.num_workers, pin_memory=True)

    # Model: ResNet18 pretrained, 1-logit head
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 1)
    model = model.to(cfg.device)

    # Class imbalance handling (pos_weight = neg/pos for abnormal=1)
    # abnormal = No Finding == 0
    y_train = (df_train["No Finding"].astype(int) == 0).astype(int).values
    pos = y_train.sum()
    neg = len(y_train) - pos
    if pos == 0:
        raise ValueError("No positive (abnormal) samples in train split.")
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32, device=cfg.device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and cfg.device.startswith("cuda")))

    # Optionally freeze backbone early (helps on small datasets)
    def set_backbone_trainable(trainable: bool):
        for name, param in model.named_parameters():
            if name.startswith("fc."):
                param.requires_grad = True
            else:
                param.requires_grad = trainable

    best_val_auc = -1.0

    for epoch in range(1, cfg.epochs + 1):
        if epoch <= cfg.freeze_backbone_epochs:
            set_backbone_trainable(False)
        else:
            set_backbone_trainable(True)

        model.train()
        running_loss = 0.0

        for x, y in dl_train:
            x = x.to(cfg.device, non_blocking=True)
            y = y.to(cfg.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(cfg.amp and cfg.device.startswith("cuda"))):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / len(ds_train)

        # Validation AUC
        val_probs, val_targets = predict_probs(model, dl_val, cfg.device)
        val_auc = compute_auc(val_probs, val_targets)

        print(f"Epoch {epoch}/{cfg.epochs} | train_loss={train_loss:.4f} | val_auc={val_auc:.4f}")

        # Save best by val AUC
        if not np.isnan(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({"model_state_dict": model.state_dict(),
                        "cfg": cfg.__dict__,
                        "best_val_auc": best_val_auc},
                       "best_resnet18_binary.pt")

    # Final test AUC using best checkpoint (if saved)
    if os.path.exists("best_resnet18_binary.pt"):
        ckpt = torch.load("best_resnet18_binary.pt", map_location=cfg.device)
        model.load_state_dict(ckpt["model_state_dict"])

    test_probs, test_targets = predict_probs(model, dl_test, cfg.device)
    test_auc = compute_auc(test_probs, test_targets)
    print(f"\nBest val_auc={best_val_auc:.4f} | test_auc={test_auc:.4f}")
    print("Saved checkpoint: best_resnet18_binary.pt")


if __name__ == "__main__":
    main()
