import torch
import torch.nn as nn
from transformers import BertModel
from config.config import CONFIG

class FinanceLLM(nn.Module):
    """Multi-task LLM with residual connections and layer norm."""
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(CONFIG.MODEL_NAME)
        self.dropout = nn.Dropout(CONFIG.DROPOUT_RATE)
        self.ln = nn.LayerNorm(768)  # BERT hidden size
        # Sentiment head
        self.sentiment_fc1 = nn.Linear(768, 256)
        self.sentiment_fc2 = nn.Linear(256, 2)
        # Q&A head
        self.qa_fc1 = nn.Linear(768, 128)
        self.qa_fc2 = nn.Linear(128, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bert_out = self.bert(input_ids, attention_mask)[1]  # [CLS] token
        x = self.ln(bert_out)
        x = self.dropout(x)
        
        # Sentiment branch with residual
        sent = torch.relu(self.sentiment_fc1(x))
        sent = self.sentiment_fc2(sent) + x[:, :2]  # Residual connection (simplified)
        
        # Q&A branch
        qa = torch.relu(self.qa_fc1(x))
        qa = self.qa_fc2(qa)
        
        return sent, qa
