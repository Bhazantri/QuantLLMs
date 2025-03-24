from transformers import AdamW
from torch.optim.lr_scheduler import StepLR
from config.config import CONFIG

def get_optimizer_and_scheduler(model) -> Tuple[AdamW, StepLR]:
    """Optimizer with weight decay and LR scheduling."""
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG.LEARNING_RATE,
        weight_decay=CONFIG.WEIGHT_DECAY
    )
    scheduler = StepLR(optimizer, step_size=CONFIG.SCHEDULER_STEP, gamma=CONFIG.SCHEDULER_GAMMA)
    return optimizer, scheduler
