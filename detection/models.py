"""
Model loading -- 5 YOLO models with caching.
"""

import os
import numpy as np
from ultralytics import YOLO
from config import cfg

_CACHE = {}


def _find_model(base_name: str) -> str:
    """Find model file: check models/ dir first, then root."""
    candidates = [
        os.path.join(cfg.MODEL_DIR, f"{base_name}.pt"),
        f"{base_name}.pt",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Model '{base_name}' not found. Searched:\n"
        + "\n".join(f"  - {c}" for c in candidates)
    )


def get_bin_models() -> tuple:
    """Returns (bin_model, material_model, size_model). Cached after first call."""
    if "bin" not in _CACHE:
        print("Loading bin detection models...")
        _CACHE["bin"] = YOLO(_find_model("best"), task="detect")
        _CACHE["material"] = YOLO(_find_model("best_material"), task="classify")
        _CACHE["size"] = YOLO(_find_model("best_size"), task="classify")

        # Warmup
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        _CACHE["bin"].predict(dummy, imgsz=640, device=cfg.DEVICE, verbose=False)
        print("Bin models loaded and cached")

    return _CACHE["bin"], _CACHE["material"], _CACHE["size"]


def get_emptying_models() -> tuple:
    """Returns (emptying_model, fullness_model). Cached after first call."""
    if "emptying" not in _CACHE:
        print("Loading emptying detection models...")
        _CACHE["emptying"] = YOLO(_find_model("best_emptying"), task="detect")
        _CACHE["fullness"] = YOLO(_find_model("best_fullness"), task="classify")
        print("Emptying models loaded and cached")

    return _CACHE["emptying"], _CACHE["fullness"]
