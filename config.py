import torch
from typing import Dict

# Advanced config with type hints and dynamic device detection
class Config:
    API_KEY: str = "your_api_key_here"  # Twelve Data API key
    MODEL_NAME: str = "bert-base-uncased"
    MAX_LEN: int = 128
    BATCH_SIZE: int = 2
    EPOCHS: int = 5
    LEARNING_RATE: float = 2e-5
    DROPOUT_RATE: float = 0.1
    WEIGHT_DECAY: float = 0.01
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    GRAD_CLIP: float = 1.0  # Gradient clipping threshold
    SCHEDULER_STEP: int = 2  # Steps for LR scheduler
    SCHEDULER_GAMMA: float = 0.1  # LR decay factor

CONFIG = Config()
