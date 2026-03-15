# train_multilabel_densenet121.py
# Multi-label CXR training on selected diseases (excluding "No Finding")
# DenseNet121 backbone

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


@dataclass
class CFG:
    manifest_path: str = "manifest_v2.parquet"
    batch_size: int = 16
    num_workers: int = 0  # Windows-safe
    epochs: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    img_size: int = 224
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    freeze_backbone_epochs: int = 1
    amp: bool = True


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class CXRMultiLabelDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_cols, transform):
        self.df = df.reset_index(drop=True)
        self.label_cols = label_cols
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = row["image_path"]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {path} | {e}")

        x = self.transform(img)
        y = torch.tensor(row[self.label_cols].values.astype(np.float32))
        return x, y


def make_transforms(img_size: int):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=3),  # milder than 7 for X-rays
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
def predict_logits(model, loader, device):
    model.eval()
    all_logits, all_targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x).detach().cpu().numpy()
        all_logits.append(logits)
        all_targets.append(y.numpy())
    return np.concatenate(all_logits, axis=0), np.concatenate(all_targets, axis=0)


def per_class_auc(y_true, y_prob, label_cols):
    aucs = {}
    for i, lab in enumerate(label_cols):
        if len(np.unique(y_true[:, i])) < 2:
            aucs[lab] = float("nan")
        else:
            aucs[lab] = roc_auc_score(y_true[:, i], y_prob[:, i])
    return aucs


def macro_auc(aucs: dict):
    vals = [v for v in aucs.values() if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    cfg = CFG()
    set_seed(cfg.seed)
    print("Device:", cfg.device)
    print("Labels:", SELECTED_LABELS)

    df = pd.read_parquet(cfg.manifest_path)

    # Checks
    req = {"image_path", "split"}
    if not req.issubset(df.columns):
        raise ValueError(f"Manifest missing: {req - set(df.columns)}")
    missing_labels = [c for c in SELECTED_LABELS if c not in df.columns]
    if missing_labels:
        raise ValueError(f"Manifest missing label columns: {missing_labels}")

    df_train = df[df["split"] == "train"].copy()
    df_val = df[df["split"] == "val"].copy()
    df_test = df[df["split"] == "test"].copy()

    # Path sanity
    first_path = df_train["image_path"].iloc[0]
    if not os.path.exists(first_path):
        raise FileNotFoundError(f"Image path not found: {first_path}")

    train_tf, eval_tf = make_transforms(cfg.img_size)

    ds_train = CXRMultiLabelDataset(df_train, SELECTED_LABELS, train_tf)
    ds_val = CXRMultiLabelDataset(df_val, SELECTED_LABELS, eval_tf)
    ds_test = CXRMultiLabelDataset(df_test, SELECTED_LABELS, eval_tf)

    dl_train = DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers, pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=cfg.batch_size, shuffle=False,
                         num_workers=cfg.num_workers, pin_memory=True)

    # Model (DenseNet121)
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(model.classifier.in_features, len(SELECTED_LABELS))
    model = model.to(cfg.device)

    # pos_weight per class (neg/pos) on train
    pos = df_train[SELECTED_LABELS].sum(axis=0).values.astype(np.float32)
    neg = (len(df_train) - pos).astype(np.float32)
    pos = np.clip(pos, 1.0, None)
    pos_weight = torch.tensor(neg / pos, dtype=torch.float32, device=cfg.device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and cfg.device.startswith("cuda")))

    def set_backbone_trainable(trainable: bool):
        for name, p in model.named_parameters():
            if name.startswith("classifier."):
                p.requires_grad = True
            else:
                p.requires_grad = trainable

    best_val_macro = -1.0

    for epoch in range(1, cfg.epochs + 1):
        set_backbone_trainable(epoch > cfg.freeze_backbone_epochs)

        model.train()
        running = 0.0

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

            running += loss.item() * x.size(0)

        train_loss = running / len(ds_train)

        # Val macro AUC
        val_logits, val_true = predict_logits(model, dl_val, cfg.device)
        val_prob = 1 / (1 + np.exp(-val_logits))
        val_aucs = per_class_auc(val_true, val_prob, SELECTED_LABELS)
        val_macro = macro_auc(val_aucs)

        print(f"Epoch {epoch}/{cfg.epochs} | train_loss={train_loss:.4f} | val_macro_auc={val_macro:.4f}")

        if not np.isnan(val_macro) and val_macro > best_val_macro:
            best_val_macro = val_macro
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "selected_labels": SELECTED_LABELS,
                    "best_val_macro_auc": best_val_macro
                },
                "best_densenet121_multilabel.pt"
            )

    # Test with best checkpoint
    ckpt = torch.load("best_densenet121_multilabel.pt", map_location=cfg.device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    test_logits, test_true = predict_logits(model, dl_test, cfg.device)
    test_prob = 1 / (1 + np.exp(-test_logits))
    test_aucs = per_class_auc(test_true, test_prob, SELECTED_LABELS)

    print("\nPer-class Test AUC:")
    for k, v in test_aucs.items():
        print(f"{k:20s} {v:.4f}")

    print("\nMacro Test AUC:", macro_auc(test_aucs))
    micro = roc_auc_score(test_true.ravel(), test_prob.ravel())
    print("Micro Test AUC:", float(micro))

    print("\nSaved checkpoint: best_densenet121_multilabel.pt")


if __name__ == "__main__":
    main()
