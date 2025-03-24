from config.config import CONFIG
from typing import Dict
import torch

def encode_text(tokenizer, text: str) -> Dict[str, torch.Tensor]:
    """Advanced encoder with position-aware attention masks."""
    encoding = tokenizer.encode_plus(
        text,
        max_length=CONFIG.MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        add_special_tokens=True,
        return_attention_mask=True,
        return_token_type_ids=True  # For segment embeddings
    )
    # Ensure proper shape and type
    encoding["input_ids"] = encoding["input_ids"].long()
    encoding["attention_mask"] = encoding["attention_mask"].float()
    encoding["token_type_ids"] = encoding["token_type_ids"].long()
    return encoding
