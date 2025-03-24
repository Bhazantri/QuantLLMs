from training.step import train_step
from training.optimizer import get_optimizer_and_scheduler
from training.loss import MultiTaskLoss
import logging

def train_model(model, train_loader, val_loader=None, epochs=CONFIG.EPOCHS):
    """Training loop with validation and logging."""
    optimizer, scheduler = get_optimizer_and_scheduler(model)
    loss_fn = MultiTaskLoss()
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            total_loss += train_step(model, batch, optimizer, scheduler, loss_fn)
        
        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}")
        
        if val_loader:
            val_loss = validate(model, val_loader, loss_fn)
            logging.info(f"Validation Loss: {val_loss:.4f}")

def validate(model, val_loader, loss_fn):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(CONFIG.DEVICE)
            attention_mask = batch["attention_mask"].to(CONFIG.DEVICE)
            labels = batch["labels"].to(CONFIG.DEVICE)
            qa_true = torch.ones_like(labels, dtype=torch.float).to(CONFIG.DEVICE) * 100.0
            sent_pred, qa_pred = model(input_ids, attention_mask)
            total_val_loss += loss_fn.compute(sent_pred, qa_pred, labels, qa_true).item()
    return total_val_loss / len(val_loader)
