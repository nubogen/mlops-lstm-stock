Project Goal — LSTM Stock Prediction MLOps Pipeline

This project aims to build a complete, production-minded Machine Learning Operations (MLOps) pipeline for time-series forecasting using real financial market data. The focus is not on perfect price prediction, but on developing a clean, modular, and reproducible ML engineering workflow that demonstrates practical MLOps competencies.

The system includes:
- A data ingestion pipeline using yfinance
- A preprocessing module that generates supervised learning sequences
- A PyTorch LSTM model architecture
- A training engine with validation and GPU acceleration (MPS on macOS)
- Full experiment tracking using MLflow (hyperparameters, metrics, model artifacts)

The expected outcome is a fully functional end-to-end pipeline that can ingest new data, retrain consistently, and log versioned experiments. This project forms a strong foundation for further MLOps features such as deployment, monitoring, scheduled retraining, and CI/CD integration.
