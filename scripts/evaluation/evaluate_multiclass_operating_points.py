import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

# ---- CONFIG ----
MANIFEST = "manifest/manifest_v2.parquet"
CKPT = input("What is the Relative Path of the model?\n").strip()
BATCH_SIZE = 32
NUM_WORKERS = 0  # Windows-safe
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224
TARGET_SPEC = 0.90
RESULTS_DIR = "results"
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
        path = row["image_path"]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        y = torch.tensor(row[self.label_cols].values.astype(np.float32))
        return x, y


def make_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def predict_probs(model, loader):
    model.eval()
    probs_list, y_list = [], []

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()
        probs_list.append(probs)
        y_list.append(y.numpy())

    return np.concatenate(probs_list, axis=0), np.concatenate(y_list, axis=0)


def threshold_for_specificity(probs, y_true, target_spec):
    neg_probs = probs[y_true == 0]
    if len(neg_probs) == 0:
        return 1.0
    return float(np.quantile(neg_probs, target_spec))


def confusion_counts(y_true, y_pred):
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return tn, fp, fn, tp


def safe_div(a, b):
    return float(a) / float(b) if b != 0 else float("nan")


def infer_model_tag(ckpt_name):
    ckpt_name = ckpt_name.lower()

    if "densenet121" in ckpt_name:
        return "densenet121"
    elif "resnet18" in ckpt_name:
        return "resnet18"
    else:
        raise ValueError(
            "Could not infer model type from checkpoint filename. "
            "Include 'resnet18' or 'densenet121' in the filename."
        )


def build_model(model_tag, num_classes):
    if model_tag == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_tag == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model tag: {model_tag}")
    return model


def load_checkpoint_state_dict(ckpt_path, device):
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    else:
        state_dict = ckpt

    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned[k[len("module."):]] = v
        else:
            cleaned[k] = v

    return cleaned


def plot_per_class_roc(test_true, test_probs, class_names, save_dir, model_tag):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 8))
    for i, label in enumerate(class_names):
        y_true = test_true[:, i].astype(int)
        y_score = test_probs[:, i]

        if len(np.unique(y_true)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f"{label} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_tag} Test ROC Curves")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_tag}_roc_all_classes.png"), dpi=300)
    plt.close()

    for i, label in enumerate(class_names):
        y_true = test_true[:, i].astype(int)
        y_score = test_probs[:, i]

        if len(np.unique(y_true)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "k--", linewidth=1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{model_tag} ROC Curve - {label}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        filename = f"{model_tag}_roc_{label.lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join(save_dir, filename), dpi=300)
        plt.close()


def plot_micro_average_roc(test_true, test_probs, save_dir, model_tag):
    os.makedirs(save_dir, exist_ok=True)

    fpr_micro, tpr_micro, _ = roc_curve(test_true.ravel(), test_probs.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr_micro, tpr_micro, linewidth=2, label=f"Micro-average AUC = {auc_micro:.3f}")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_tag} Micro-Average ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_tag}_roc_micro_average.png"), dpi=300)
    plt.close()


def plot_auc_bar_chart(report_df, save_dir, model_tag):
    os.makedirs(save_dir, exist_ok=True)

    plot_df = report_df.sort_values("test_auc", ascending=True)

    plt.figure(figsize=(9, 6))
    plt.barh(plot_df["label"], plot_df["test_auc"])
    plt.xlabel("Test ROC-AUC")
    plt.ylabel("Pathology")
    plt.title(f"{model_tag} Test AUC by Class")
    plt.xlim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_tag}_auc_bar_chart.png"), dpi=300)
    plt.close()


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    ckpt_path = os.path.normpath(CKPT)
    ckpt_name = os.path.basename(ckpt_path).lower()
    model_tag = infer_model_tag(ckpt_name)

    print(f"Detected model: {model_tag}")

    df = pd.read_parquet(MANIFEST)

    df_val = df[df["split"] == "val"].copy()
    df_test = df[df["split"] == "test"].copy()

    tfm = make_transform(IMG_SIZE)

    val_loader = DataLoader(
        CXRMultiLabelDataset(df_val, SELECTED_LABELS, tfm),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )

    test_loader = DataLoader(
        CXRMultiLabelDataset(df_test, SELECTED_LABELS, tfm),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda"),
    )

    model = build_model(model_tag, len(SELECTED_LABELS))
    model.to(DEVICE)

    state_dict = load_checkpoint_state_dict(ckpt_path, DEVICE)
    model.load_state_dict(state_dict)

    val_probs, val_true = predict_probs(model, val_loader)
    test_probs, test_true = predict_probs(model, test_loader)

    rows = []
    for i, label in enumerate(SELECTED_LABELS):
        v_y = val_true[:, i].astype(int)
        v_p = val_probs[:, i]
        t_y = test_true[:, i].astype(int)
        t_p = test_probs[:, i]

        val_auc = roc_auc_score(v_y, v_p) if len(np.unique(v_y)) > 1 else float("nan")
        test_auc = roc_auc_score(t_y, t_p) if len(np.unique(t_y)) > 1 else float("nan")

        thr = threshold_for_specificity(v_p, v_y, TARGET_SPEC)

        t_pred = (t_p >= thr).astype(int)
        tn, fp, fn, tp = confusion_counts(t_y, t_pred)

        spec = safe_div(tn, tn + fp)
        sens = safe_div(tp, tp + fn)
        prec = safe_div(tp, tp + fp)

        rows.append({
            "label": label,
            "val_pos": int(v_y.sum()),
            "test_pos": int(t_y.sum()),
            "thr@val_spec": thr,
            "val_auc": val_auc,
            "test_auc": test_auc,
            "test_spec": spec,
            "test_sens": sens,
            "test_prec": prec,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
        })

    out = pd.DataFrame(rows).sort_values("test_auc", ascending=False)

    macro_auc = roc_auc_score(test_true, test_probs, average="macro")
    micro_auc = roc_auc_score(test_true, test_probs, average="micro")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.max_rows", 200)

    print(f"\nOperating point: thresholds chosen for ~{int(TARGET_SPEC * 100)}% specificity on VAL\n")
    print(out[
        ["label", "val_pos", "test_pos", "thr@val_spec", "val_auc", "test_auc",
         "test_spec", "test_sens", "test_prec", "TN", "FP", "FN", "TP"]
    ].to_string(index=False))

    print(f"\nMacro ROC-AUC (test): {macro_auc:.4f}")
    print(f"Micro ROC-AUC (test): {micro_auc:.4f}")

    report_path = os.path.join(RESULTS_DIR, f"{model_tag}_operating_points_report.csv")
    out.to_csv(report_path, index=False)
    print(f"\nSaved: {report_path}")

    summary_df = pd.DataFrame([{
        "model": model_tag,
        "macro_auc": macro_auc,
        "micro_auc": micro_auc,
        "target_specificity": TARGET_SPEC,
        "num_val_samples": len(df_val),
        "num_test_samples": len(df_test),
    }])
    summary_path = os.path.join(RESULTS_DIR, f"{model_tag}_summary_metrics.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    plot_per_class_roc(test_true, test_probs, SELECTED_LABELS, RESULTS_DIR, model_tag)
    print(f"Saved: {os.path.join(RESULTS_DIR, f'{model_tag}_roc_all_classes.png')}")
    print(f"Saved: individual per-class ROC curve PNGs with prefix '{model_tag}_'")

    plot_micro_average_roc(test_true, test_probs, RESULTS_DIR, model_tag)
    print(f"Saved: {os.path.join(RESULTS_DIR, f'{model_tag}_roc_micro_average.png')}")

    plot_auc_bar_chart(out, RESULTS_DIR, model_tag)
    print(f"Saved: {os.path.join(RESULTS_DIR, f'{model_tag}_auc_bar_chart.png')}")


if __name__ == "__main__":
    main()
