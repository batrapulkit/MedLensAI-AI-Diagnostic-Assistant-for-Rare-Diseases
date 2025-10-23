import torch
import numpy as np
from pathlib import Path
from torchvision import transforms
from core.data_utils import load_image_any
from sklearn.metrics.pairwise import cosine_similarity

def compute_class_prototypes(support_dir, embedder):
    support_dir = Path(support_dir)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))
    ])

    prototypes = {}
    for cls_folder in support_dir.iterdir():
        if not cls_folder.is_dir():
            continue
        cls_name = cls_folder.name
        img_paths = list(cls_folder.glob("*"))
        embeddings = []
        for img_path in img_paths:
            try:
                pil_img = load_image_any(img_path)
                tensor = transform(pil_img).unsqueeze(0)
                with torch.no_grad():
                    _, emb = embedder(tensor)
                    emb = emb.cpu().numpy().reshape(-1)
                    emb = emb / np.linalg.norm(emb)
                    embeddings.append(emb)
            except Exception as e:
                print(f"⚠️ Skipping {img_path}: {e}")
        if embeddings:
            prototype = np.mean(embeddings, axis=0)
            prototype = prototype / np.linalg.norm(prototype)
            prototypes[cls_name] = prototype
            print(f"✅ {cls_name}: {len(embeddings)} images loaded.")
    return prototypes

def infer_diagnosis(prototypes, query_embedding):
    class_names = list(prototypes.keys())
    proto_vectors = np.stack([prototypes[c] for c in class_names], axis=0)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    similarities = cosine_similarity(proto_vectors, query_embedding.reshape(1, -1)).reshape(-1)
    dists = 1 - similarities
    result_dict = dict(zip(class_names, dists.tolist()))
    pred_class = class_names[int(np.argmin(dists))]
    return pred_class, result_dict
