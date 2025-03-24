# FinanceLLMAgent: A Modular Deep Learning Framework for Financial Analysis

## Project Description
`FinanceLLMAgent` is an advanced, research-grade deep learning framework designed for financial analysis, built with a modular, folder-based architecture inspired by George Hotz’s minimalist and iterative coding philosophy. This project eschews data normalization to preserve raw financial signals, leveraging a multi-task BERT-based Large Language Model (LLM) to perform sentiment analysis and question-answering (Q&A) on financial data. It integrates real-time stock data via the Twelve Data API, employs cutting-edge deep learning techniques, and is structured into 15+ self-contained segments for maximum flexibility and scalability.

The framework is engineered for technical depth, featuring multi-task learning with residual connections, gradient clipping, learning rate scheduling, uncertainty estimation, and custom tokenization for financial contexts. It’s a production-ready yet hackable codebase, ideal for researchers exploring AI-driven finance, developers building financial tools, or enthusiasts seeking a Hotz-style modular project to dissect and extend.

## Key Features
- **Raw Data Processing**: Operates on unnormalized stock prices and volumes, fetched live from Twelve Data API, preserving original data distributions.
- **Multi-Task LLM**: A BERT-based model with dual heads: a classification head for sentiment (positive/negative) and a regression head for Q&A (e.g., price predictions).
- **Modular Design**: 15+ independent files across folders (`config`, `data`, `model`, `training`, `inference`, `agent`), each encapsulating a specific functionality for easy testing and iteration.
- **Advanced Techniques**: Includes residual connections, layer normalization, gradient clipping, StepLR scheduling, entropy-based uncertainty, and custom financial tokenization.
- **Profiling & Logging**: Built-in performance profiling and detailed logging for debugging and optimization.

## Directory Structure & Technical Segments
The project is split into a folder-based structure with 15+ segments, each designed to be standalone yet cohesive:

1. **`config/config.py`**: Centralized configuration with hyperparameters (e.g., `LEARNING_RATE=2e-5`, `GRAD_CLIP=1.0`), dynamic device detection, and scheduler settings.
2. **`data/fetcher.py`**: Robust stock data fetcher with retry logic (3 attempts), timeout handling, and pandas-based raw data parsing from Twelve Data API.
3. **`data/dataset.py`**: Custom PyTorch Dataset with optional text augmentation (e.g., synonym substitution) for financial text, supporting dynamic batching.
4. **`model/tokenizer.py`**: BERT tokenizer enhanced with finance-specific tokens (e.g., "AAPL", "$"), ensuring domain-aware tokenization.
5. **`model/encoder.py`**: Advanced text encoder with position-aware attention masks and token type IDs for segment embeddings, optimized for BERT input.
6. **`model/llm.py`**: Multi-task LLM with a BERT backbone, residual connections in the sentiment head, layer normalization, and separate FC layers (768→256→2 for sentiment, 768→128→1 for Q&A).
7. **`model/builder.py`**: Model constructor with Xavier initialization for linear layers, ensuring stable training convergence.
8. **`training/loss.py`**: Multi-task loss combining weighted CrossEntropy (sentiment, 0.7) and MSE (Q&A, 0.3) for balanced optimization.
9. **`training/optimizer.py`**: AdamW optimizer with weight decay (0.01) and StepLR scheduler (step=2, gamma=0.1) for adaptive learning rate decay.
10. **`training/step.py`**: Training step with gradient clipping (max norm 1.0), multi-task loss computation, and scheduler integration.
11. **`training/loop.py`**: Full training loop with validation support, epoch-wise logging, and loss aggregation for monitoring.
12. **`inference/sentiment.py`**: Sentiment inference with softmax probabilities and entropy-based uncertainty estimation for confidence assessment.
13. **`inference/qa.py`**: Q&A inference with context-aware embedding fusion, sigmoid-scaled regression output (0-1000 range), and financial interpretability.
14. **`agent/logic.py`**: Dynamic agent logic with input dispatching (sentiment vs. Q&A), integrating data fetching and inference pipelines.
15. **`main.py`**: Main runner with sample dataset, training execution, inference testing, and cProfile-based performance profiling.

## Technical Specifications
- **Model Architecture**:
  - Backbone: BERT (`bert-base-uncased`, 768 hidden size).
  - Sentiment Head: 768 → 256 (ReLU) → 2 (linear) with residual skip.
  - Q&A Head: 768 → 128 (ReLU) → 1 (linear).
  - Regularization: Dropout (0.1), LayerNorm.
- **Training**:
  - Optimizer: AdamW (lr=2e-5, weight_decay=0.01).
  - Scheduler: StepLR (step_size=2, gamma=0.1).
  - Loss: `0.7 * CrossEntropy + 0.3 * MSE`.
  - Gradient Clipping: Max norm 1.0.
  - Epochs: 5 (configurable).
- **Inference**:
  - Sentiment: Softmax with entropy uncertainty.
  - Q&A: Sigmoid-scaled regression (0-1000 range).
- **Data**: Raw stock data (e.g., OHLCV) from Twelve Data API, no normalization.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Bhazantri/FinanceLLMAgent.git
   cd FinanceLLMAgent
