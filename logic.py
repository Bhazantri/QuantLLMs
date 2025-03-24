from data.fetcher import FETCHER
from inference.sentiment import predict_sentiment
from inference.qa import answer_query
from typing import Optional

class FinanceAgent:
    """Dynamic agent with input dispatching."""
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def process(self, user_input: str, symbol: Optional[str] = None) -> str:
        context = FETCHER.fetch(symbol).to_string() if symbol else "No data."
        
        if "?" in user_input:
            return answer_query(self.model, self.tokenizer, user_input, context)
        else:
            sentiment, conf, uncert = predict_sentiment(self.model, self.tokenizer, user_input)
            return f"Sentiment: {sentiment} (Conf: {conf:.2f}, Uncertainty: {uncert:.2f})"
