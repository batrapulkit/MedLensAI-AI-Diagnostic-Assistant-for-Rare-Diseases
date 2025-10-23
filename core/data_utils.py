from PIL import Image
import numpy as np
import pydicom
from pathlib import Path

def load_image_any(path):
    path = Path(path)
    suf = path.suffix.lower()
    if suf in [".jpg", ".jpeg", ".png"]:
        return Image.open(path).convert("RGB")
    if suf == ".dcm":
        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        arr = (arr * 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr]*3, axis=-1)
        return Image.fromarray(arr)
    raise ValueError(f"Unsupported file type: {path}")
