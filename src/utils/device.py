# src/utils/device.py
import torch

def get_best_device(preferred="cuda"):
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    elif preferred == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
