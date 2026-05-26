import torch
import torch.nn as nn
from tqdm import tqdm


def mse_loss(pred, target):
    # pred: (B, 1)  target: (B,)
    return nn.functional.mse_loss(pred.squeeze(1), target)


def train_one_epoch(model, loader, optimiser, scaler, device):
    model.train()
    total_loss = 0.0
    for imgs, labels in tqdm(loader, desc='  train', leave=False):
        imgs   = imgs.to(device)
        labels = labels.to(device)
        optimiser.zero_grad()
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            preds = model(imgs)
            loss  = mse_loss(preds, labels)
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


