import torch
import numpy as np
import cv2

def generate_gradcam(embedder, tensor):
    tensor = tensor.clone().detach().requires_grad_(True).to(embedder.device)
    emb = embedder.model(tensor)
    score = emb.sum()
    score.backward()

    grad = tensor.grad.detach().cpu().numpy().mean(axis=1)[0]
    grad = np.maximum(grad, 0)
    grad = (grad - grad.min()) / (grad.max() - grad.min() + 1e-8)
    grad = cv2.resize(grad, (224, 224))
    return grad

def overlay_heatmap(img, heatmap, alpha=0.5):
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    img = np.uint8(img)
    overlay = np.uint8(img * (1 - alpha) + heatmap_colored * alpha)
    return overlay
