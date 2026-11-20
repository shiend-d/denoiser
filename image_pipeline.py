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

def add_gaussian_noise(image: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """Create one deterministic noisy RGB input shared by every execution mode."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def process_slice(
    image: np.ndarray,
    nlm_search_window: int,
    nlm_h: float,
    nlm_h_color: float,
) -> np.ndarray:
    """Denoise one RGB tile with OpenCV's colored Non-Local Means filter."""
    # OpenCV's colored NLM expects BGR; the rest of the application uses RGB.
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    denoised = cv2.fastNlMeansDenoisingColored(
        bgr, None, nlm_h, nlm_h_color, 7, nlm_search_window,
    )
    return cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)

def save_image(path: str | Path, image: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="RGB").save(path)
