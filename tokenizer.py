from transformers import BertTokenizer
from config.config import CONFIG
import logging

def get_tokenizer() -> BertTokenizer:
    """Tokenizer with custom financial vocab injection."""
    tokenizer = BertTokenizer.from_pretrained(CONFIG.MODEL_NAME)
    # Add finance-specific tokens (e.g., ticker symbols)
    new_tokens = ["AAPL", "TSLA", "GOOGL", "$"]
    num_added = tokenizer.add_tokens(new_tokens)
    logging.info(f"Added {num_added} financial tokens to tokenizer")
    return tokenizer
