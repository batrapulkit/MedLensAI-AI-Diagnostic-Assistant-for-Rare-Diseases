# 🔬 MedLensAI – AI Diagnostic Assistant for Rare Diseases

MedLensAI is an AI-powered diagnostic assistant designed to help clinicians identify **rare diseases** from medical images using **few-shot learning** and **explainable AI** techniques.

Traditional AI models require large, annotated datasets — something rare diseases simply don't have. MedLensAI tackles this with cutting-edge few-shot learning methods and pretrained vision transformers, enabling generalization to unseen diseases with minimal data.

---

## 🧠 Features

- ✅ **Few-Shot Learning Support**  
  Learn rare disease patterns from as few as 1–5 examples.

- ✅ **Pretrained Vision Transformers**  
  Uses models like ViT and CLIP for strong prior visual knowledge.

- ✅ **Medical Image Compatibility**  
  Works with common formats including DICOM, PNG, and JPEG.

- ✅ **Explainable AI**  
  Integrates Grad-CAM and SHAP to highlight regions influencing the model’s predictions — ensuring interpretability for clinicians.

- ✅ **Modular & Extensible**  
  Built with flexibility for research, clinical testing, or educational use.

---

## 📸 Sample Use Cases

- Diagnosing **rare genetic disorders** from facial or radiographic images  
- Identifying **rare cancers** using histopathology slides  
- Supporting under-resourced clinics with limited diagnostic specialists

---

## ⚙️ How It Works

1. **Image Input**  
   Upload or pass in a medical image for analysis.

2. **Few-Shot Learning Inference**  
   The model compares input with few labeled support examples using a metric-based approach (e.g., Prototypical Networks or Siamese Networks).

3. **Prediction + Explanation**  
   Returns a predicted class (if match found) with a visual heatmap showing the areas most influential in the model's decision.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- torchvision
- scikit-learn
- opencv-python
- matplotlib
- pydicom (for DICOM support)

## Contribution
I welcome contributions! Feel free to open issues, submit pull requests, or suggest ideas to improve MedLensAI.
