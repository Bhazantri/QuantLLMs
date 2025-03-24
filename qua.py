import torch
from config.config import CONFIG
from model.encoder import encode_text

def answer_query(model, tokenizer, question: str, context: str) -> str:
    """Q&A with context-aware embedding fusion."""
    model.eval()
    combined = f"{question} [SEP] {context}"
    enc = encode_text(tokenizer, combined)
    with torch.no_grad():
        _, qa_out = model(enc["input_ids"].to(CONFIG.DEVICE), enc["attention_mask"].to(CONFIG.DEVICE))
        # Scale output to realistic financial range (heuristic)
        scaled_value = torch.sigmoid(qa_out) * 1000  # E.g., stock price range
        return f"Predicted value: {scaled_value.item():.2f}"
