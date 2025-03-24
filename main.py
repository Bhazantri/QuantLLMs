import torch
from torch.utils.data import DataLoader
from config.config import CONFIG
from data.dataset import FinanceDataset
from model.tokenizer import get_tokenizer
from model.builder import build_model
from training.loop import train_model
from agent.logic import FinanceAgent
import logging
import cProfile

def main():
    logging.basicConfig(level=logging.INFO)
    tokenizer = get_tokenizer()
    model = build_model()

    # Sample data
    texts = ["Stocks are soaring!", "Market is doomed."]
    labels = [1, 0]
    dataset = FinanceDataset(texts, labels, tokenizer)
    loader = DataLoader(dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True)

    # Train with profiling
    profiler = cProfile.Profile()
    profiler.enable()
    train_model(model, loader)
    profiler.disable()
    profiler.dump_stats("train_profile.prof")

    # Agent inference
    agent = FinanceAgent(model, tokenizer)
    test_inputs = [
        ("What’s AAPL’s trend?", "AAPL"),
        ("The economy is thriving!", None)
    ]
    for inp, sym in test_inputs:
        logging.info(f"Input: {inp}")
        logging.info(f"Output: {agent.process(inp, sym)}")

if __name__ == "__main__":
    main()
