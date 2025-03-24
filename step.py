from config.config import CONFIG
from training.loss import MultiTaskLoss

def train_step(model, batch, optimizer, scheduler, loss_fn: MultiTaskLoss):
    """Training step with gradient clipping and multi-task loss."""
    optimizer.zero_grad()
    input_ids = batch["input_ids"].to(CONFIG.DEVICE)
    attention_mask = batch["attention_mask"].to(CONFIG.DEVICE)
    labels = batch["labels"].to(CONFIG.DEVICE)
    
    # Dummy QA target (e.g., price trend)
    qa_true = torch.ones_like(labels, dtype=torch.float).to(CONFIG.DEVICE) * 100.0
    
    sentiment_logits, qa_out = model(input_ids, attention_mask)
    loss = loss_fn.compute(sentiment_logits, qa_out, labels, qa_true)
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.GRAD_CLIP)
    optimizer.step()
    scheduler.step()
    return loss.item()
