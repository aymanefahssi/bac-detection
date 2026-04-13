"""
Configuration loader.

Priority: environment variable > settings.yaml > built-in default.
Edit settings.yaml for easy maintenance.
"""

import os
import yaml

# ---------------------------------------------------------------------------
# Load settings.yaml
# ---------------------------------------------------------------------------
_SETTINGS_PATH = os.path.join(os.path.dirname(__file__), "settings.yaml")

def _load_yaml() -> dict:
    if os.path.exists(_SETTINGS_PATH):
        with open(_SETTINGS_PATH, "r") as f:
            return yaml.safe_load(f) or {}
    return {}

_Y = _load_yaml()

def _get(env_key: str, yaml_path: str, default):
    """Resolve a value: env var wins, then yaml, then default."""
    env = os.getenv(env_key)
    if env is not None:
        return type(default)(env)
    # Walk dotted yaml path (e.g. "system1.confidence")
    node = _Y
    for part in yaml_path.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return default
    return type(default)(node)


# ---------------------------------------------------------------------------
# GPU setup
# ---------------------------------------------------------------------------
def _setup_gpu() -> bool:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            return True
        print("CUDA not available, using CPU")
        return False
    except ImportError:
        print("PyTorch not installed, using CPU")
        return False

GPU_AVAILABLE = _setup_gpu()


# ---------------------------------------------------------------------------
# Config object
# ---------------------------------------------------------------------------
class Config:
    # --- Device ---
    DEVICE: str = _get("DEVICE", "device", "0" if GPU_AVAILABLE else "cpu")
    HALF_PRECISION: bool = GPU_AVAILABLE

    # --- Models ---
    MODEL_DIR: str = _get("MODEL_DIR", "model_dir", "models")

    # --- System 1: Bin Detection ---
    YOLO_CONFIDENCE: float = _get("YOLO_CONFIDENCE", "system1.confidence", 0.40)
    FRAME_INTERVAL: float = _get("FRAME_INTERVAL", "system1.frame_interval", 1.5)

    # --- System 2: Emptying Detection ---
    EMPTYING_CONFIDENCE: float = _get("EMPTYING_CONFIDENCE", "system2.confidence", 0.15)
    EMPTYING_FRAME_INTERVAL: float = _get("EMPTYING_FRAME_INTERVAL", "system2.frame_interval", 0.5)

    # --- Emptying State Machine ---
    CONSECUTIVE_NORMAL_FRAMES_TO_START: int = _get(
        "CONSECUTIVE_NORMAL_FRAMES_TO_START", "emptying.consecutive_normal_to_start", 2)
    CONSECUTIVE_NORMAL_FRAMES: int = _get(
        "CONSECUTIVE_NORMAL_FRAMES", "emptying.consecutive_normal_to_end", 3)
    CONSECUTIVE_EMPTYING_FRAMES: int = _get(
        "CONSECUTIVE_EMPTYING_FRAMES", "emptying.consecutive_emptying_to_start", 1)
    MAX_EVENT_DURATION: float = _get(
        "MAX_EVENT_DURATION", "emptying.max_event_duration", 15.0)
    NO_BINS_TIMEOUT: float = _get(
        "NO_BINS_TIMEOUT", "emptying.no_bins_timeout", 5.0)
    EMPTYING_COOLDOWN: float = _get(
        "EMPTYING_COOLDOWN", "emptying.cooldown", 3.0)
    EMPTYING_FRAMES_TO_CAPTURE: int = _get(
        "EMPTYING_FRAMES_TO_CAPTURE", "emptying.frames_to_capture", 3)


cfg = Config()
