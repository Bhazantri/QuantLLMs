import torch.nn as nn
from typing import Tuple

class MultiTaskLoss:
    """Combined loss for sentiment and Q&A tasks."""
    def __init__(self, sentiment_weight: float = 0.7, qa_weight: float = 0.3):
        self.sentiment_loss = nn.CrossEntropyLoss()
        self.qa_loss = nn.MSELoss()
        self.sw = sentiment_weight
        self.qw = qa_weight

    def compute(self, sentiment_pred: torch.Tensor, qa_pred: torch.Tensor, 
                sentiment_true: torch.Tensor, qa_true: torch.Tensor) -> torch.Tensor:
        s_loss = self.sentiment_loss(sentiment_pred, sentiment_true)
        q_loss = self.qa_loss(qa_pred, qa_true)
        return self.sw * s_loss + self.qw * q_loss
