import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import argparse
import torchvision.transforms as T
from picarnet import PiCarNet
from pathlib import Path

# Paths 
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
REPO_DIR     = os.path.dirname(os.path.dirname(os.path.dirname(BASE_DIR)))
DATA_PATH    = os.path.join(REPO_DIR, "data")
MODELS_DIR   = os.path.join(BASE_DIR, "outputs", "models")
HEATMAPS_DIR = os.path.join(REPO_DIR, "heatmaps")
os.makedirs(HEATMAPS_DIR, exist_ok=True)

# Config 
CONFIG = {
    "model_name"   : "crop_oversample_corners_best_model.pth",
    "img_dir"      : os.path.join(DATA_PATH, "training_images"),
    "n_images"     : 10,
    "skip"         : 0,
    "crop_top"     : 0.30,
    "crop_bottom"  : 0,
    "img_h"        : 120,
    "img_w"        : 160,
    "single_output": True,    # True = angle only, False = angle + speed
}

# Transforms 
def build_transform(crop_top, crop_bottom, img_h, img_w):
    steps = [T.ToPILImage()]
    if crop_top > 0 or crop_bottom > 0:
        steps.append(T.Lambda(lambda img: T.functional.crop(
            img,
            top    = int(img.height * crop_top),
            left   = 0,
            height = int(img.height * (1 - crop_top - crop_bottom)),
            width  = img.width,
        )))
    steps += [
        T.Resize((img_h, img_w)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ]
    return T.Compose(steps)


def crop_np(img_np, crop_top, crop_bottom):
    h      = img_np.shape[0]
    top    = int(h * crop_top)
    bottom = int(h * (1 - crop_bottom)) if crop_bottom > 0 else h
    return img_np[top:bottom, :]

# Grad-CAM 
def get_gradcam(model, image_tensor, output_idx=0):
    model.eval()
    gradients, activations = [], []

    target_layer = model.backbone.features[-1]
    fh = target_layer.register_forward_hook(
        lambda m, i, o: activations.append(o))
    bh = target_layer.register_full_backward_hook(
        lambda m, gi, go: gradients.append(go[0]))

    output = model(image_tensor)
    model.zero_grad()
    output[0, output_idx].backward()

    fh.remove()
    bh.remove()

    grads   = gradients[0].squeeze(0)
    acts    = activations[0].squeeze(0)
    weights = grads.mean(dim=(1, 2))
    heatmap = F.relu((weights[:, None, None] * acts).sum(dim=0))
    heatmap = heatmap / (heatmap.max() + 1e-8)
    return heatmap.detach().cpu().numpy()


def overlay_gradcam(img_bgr, heatmap):
    heatmap = cv2.resize(heatmap, (img_bgr.shape[1], img_bgr.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    return cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)

