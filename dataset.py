from torch.utils.data import Dataset
import torch
from model.encoder import encode_text
from typing import List
import numpy as np

class FinanceDataset(Dataset):
    """Dataset with data augmentation for financial text."""
    def __init__(self, texts: List[str], labels: List[int], tokenizer, augment: bool = True):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.augment = augment

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        if self.augment and np.random.rand() > 0.5:
            text = self._augment_text(text)  # Simple synonym swap (placeholder)
        enc = encode_text(self.tokenizer, text)
        return {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

    def _augment_text(self, text: str) -> str:
        # Placeholder: Swap "good" with "great", etc.
        return text.replace("good", "great") if "good" in text else text
