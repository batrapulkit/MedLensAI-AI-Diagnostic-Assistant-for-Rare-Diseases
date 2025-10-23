import streamlit as st
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from core.embedder import load_model
from core.data_utils import load_image_any
from core.model import compute_class_prototypes, infer_diagnosis
from core.explainability import generate_gradcam, overlay_heatmap
from pathlib import Path

st.set_page_config(page_title="MedLensAI", page_icon="🩺")
st.title("🩺 MedLensAI – AI Diagnostic Assistant for Rare Diseases")
st.caption("Few-shot + Explainable AI Prototype")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

@st.cache_resource
def get_model():
    return load_model(device="cpu")

embedder = get_model()
support_dir = "data/support"

if not Path(support_dir).exists():
    st.error("❌ Support data not found. Please add folders under data/support/.")
    st.stop()

st.info("📁 Computing class prototypes...")
prototypes = compute_class_prototypes(support_dir, embedder)
st.success(f"✅ Loaded {len(prototypes)} disease classes: {list(prototypes.keys())}")

uploaded = st.file_uploader("📤 Upload a medical image to analyze", type=["jpg", "jpeg", "png", "dcm"])

if uploaded:
    query_path = Path(f"temp_{uploaded.name}")
    with open(query_path, "wb") as f:
        f.write(uploaded.read())
    pil_img = load_image_any(query_path)
    st.image(pil_img, caption="Uploaded Image", width=300)

    tensor = transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        _, emb = embedder(tensor)
        emb = emb.numpy().reshape(-1)

    pred_class, distances = infer_diagnosis(prototypes, emb)
    st.subheader(f"🧠 Prediction: **{pred_class}**")

    st.markdown("### 📊 Distance Scores")
    st.json({k: round(v, 3) for k, v in distances.items()})

    st.markdown("### 🔍 Grad-CAM Visualization")
    try:
        heatmap = generate_gradcam(embedder, tensor)
        np_img = np.array(pil_img.resize((224, 224)))
        overlay = overlay_heatmap(np_img, heatmap)
        st.image([pil_img, Image.fromarray(overlay)],
                 caption=["Original", "Model Focus"],
                 width=300)
    except Exception as e:
        st.warning(f"Grad-CAM could not be generated: {e}")

st.caption("⚠️ Research prototype – not for clinical use.")
