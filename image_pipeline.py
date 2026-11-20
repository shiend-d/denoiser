from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

def read_image(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()

def checksum(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:16]

def process_slice(
    image: np.ndarray,
    scale: float,
    operation: str = "upscale",
    nlm_search_window: int = 21,
) -> np.ndarray:
    if operation == "nlm":
        # OpenCV's colored NLM expects BGR; the rest of the application uses RGB.
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        denoised = cv2.fastNlMeansDenoisingColored(
            bgr, None, 3, 3, 7, nlm_search_window,
        )
        return cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
    if operation != "upscale":
        raise ValueError(f"unsupported image operation: {operation}")

    height, width = image.shape[:2]
    target = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_LANCZOS4)
    lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = cv2.cvtColor(cv2.merge((clahe.apply(l_channel), a_channel, b_channel)), cv2.COLOR_LAB2RGB)
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
    return np.clip(cv2.addWeighted(enhanced, 1.35, blurred, -0.35, 0), 0, 255).astype(np.uint8, copy=False)

def save_image(path: str | Path, image: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(path)
