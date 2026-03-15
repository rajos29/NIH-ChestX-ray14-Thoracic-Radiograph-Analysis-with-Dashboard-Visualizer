from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
import matplotlib.cm as cm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

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

DISPLAY_LABELS = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Consolidation": "Consolidation",
    "Effusion": "Effusion",
    "Emphysema": "Emphysema",
    "Infiltration": "Infiltration",
    "Mass": "Mass",
    "Nodule": "Nodule",
    "Pleural_Thickening": "Pleural Thickening",
    "Pneumothorax": "Pneumothorax",
}

DISPLAY_TO_INTERNAL = {v: k for k, v in DISPLAY_LABELS.items()}


def make_transform(img_size: int = IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def disable_inplace_relu(module):
    for child in module.children():
        if isinstance(child, torch.nn.ReLU):
            child.inplace = False
        disable_inplace_relu(child)


def load_resnet_model(ckpt_path: str):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(SELECTED_LABELS))

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    disable_inplace_relu(model)
    model.to(DEVICE)
    model.eval()
    return model


def load_densenet_model(ckpt_path: str):
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, len(SELECTED_LABELS))

    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    disable_inplace_relu(model)
    model.to(DEVICE)
    model.eval()
    return model


def _choose_target_layer(model_name: str, model):
    if model_name == "ResNet18":
        return model.layer4[-1]
    elif model_name == "DenseNet121":
        return model.features.denseblock4
    raise ValueError(f"Unsupported model name: {model_name}")


def _normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = cam - cam.min()
    max_val = cam.max()
    if max_val > 0:
        cam = cam / max_val
    return cam


def _overlay_heatmap_on_image(
    original_rgb: np.ndarray,
    cam_map: np.ndarray,
    alpha: float = 0.42,
) -> Image.Image:
    heatmap = cm.jet(cam_map)[..., :3]
    overlay = (1 - alpha) * original_rgb + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)
    overlay_uint8 = (overlay * 255).astype(np.uint8)
    return Image.fromarray(overlay_uint8)


def generate_gradcam_overlay(
    model,
    model_name: str,
    image_path: str,
    target_label: str,
) -> Tuple[Image.Image, np.ndarray]:
    """
    Returns:
        overlay_img: PIL image with Grad-CAM overlay
        cam_map: normalized heatmap as HxW numpy array
    """
    if target_label not in SELECTED_LABELS:
        raise ValueError(f"Unknown target label: {target_label}")

    target_layer = _choose_target_layer(model_name, model)
    activations = {}
    gradients = {}

    def forward_hook(module, inp, out):
        activations["value"] = out.detach().clone()

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach().clone()

    f_handle = target_layer.register_forward_hook(forward_hook)
    b_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        pil_img = Image.open(image_path).convert("RGB")
        original_resized = pil_img.resize((IMG_SIZE, IMG_SIZE))
        original_rgb = np.asarray(original_resized).astype(np.float32) / 255.0

        tfm = make_transform(IMG_SIZE)
        x = tfm(pil_img).unsqueeze(0).to(DEVICE)

        model.zero_grad(set_to_none=True)
        logits = model(x)

        class_idx = SELECTED_LABELS.index(target_label)
        score = logits[:, class_idx].sum()
        score.backward()

        acts = activations["value"]
        grads = gradients["value"]

        if acts.ndim != 4 or grads.ndim != 4:
            raise ValueError(
                f"Grad-CAM expected 4D activations/gradients but got "
                f"acts={tuple(acts.shape)}, grads={tuple(grads.shape)}"
            )

        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = torch.nn.functional.interpolate(
            cam,
            size=(IMG_SIZE, IMG_SIZE),
            mode="bilinear",
            align_corners=False,
        )

        cam_map = cam[0, 0].cpu().numpy()
        cam_map = _normalize_cam(cam_map)

        overlay_img = _overlay_heatmap_on_image(original_rgb, cam_map)
        return overlay_img, cam_map

    finally:
        f_handle.remove()
        b_handle.remove()
