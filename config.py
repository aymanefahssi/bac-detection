"""
Minimal configuration for standalone detection.
"""

import os


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


class Config:
    # Device
    DEVICE: str = "0" if GPU_AVAILABLE else "cpu"
    HALF_PRECISION: bool = GPU_AVAILABLE

    # Detection (video mode)
    YOLO_CONFIDENCE: float = float(os.getenv("YOLO_CONFIDENCE", "0.40"))
    EMPTYING_CONFIDENCE: float = float(os.getenv("EMPTYING_CONFIDENCE", "0.40"))
    FRAME_INTERVAL: float = float(os.getenv("FRAME_INTERVAL", "1.5"))

    # Models
    MODEL_DIR: str = os.getenv("MODEL_DIR", "models")

    # Emptying state machine
    CONSECUTIVE_NORMAL_FRAMES_TO_START: int = 4
    CONSECUTIVE_NORMAL_FRAMES: int = 6
    CONSECUTIVE_EMPTYING_FRAMES: int = 4
    MAX_EVENT_DURATION: float = 15.0
    NO_BINS_TIMEOUT: float = 5.0
    EMPTYING_COOLDOWN: float = 3.0
    EMPTYING_FRAMES_TO_CAPTURE: int = 3


cfg = Config()
