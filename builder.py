from model.llm import FinanceLLM
from config.config import CONFIG
import torch.nn.init as init

def build_model() -> FinanceLLM:
    """Model builder with custom weight initialization."""
    model = FinanceLLM().to(CONFIG.DEVICE)
    # Xavier initialization for linear layers
    for name, param in model.named_parameters():
        if "fc" in name and "weight" in name:
            init.xavier_uniform_(param.data)
        elif "fc" in name and "bias" in name:
            init.constant_(param.data, 0.0)
    return model
