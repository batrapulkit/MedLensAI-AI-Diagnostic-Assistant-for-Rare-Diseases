import torch
import timm

class MedEmbedder(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()
        # Simple, stable pretrained Vision Transformer
        self.model = timm.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
        self.model.to(device).eval()
        self.device = device

    def forward(self, x):
        # Standard forward: get embeddings only (no Grad-CAM hooks)
        x = x.to(self.device)
        with torch.no_grad():
            emb = self.model(x)
        emb = emb / torch.norm(emb, dim=1, keepdim=True)
        return None, emb


def load_model(device="cpu"):
    return MedEmbedder(device=device)
