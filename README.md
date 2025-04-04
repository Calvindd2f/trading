# Trading System

A sophisticated trading system with multiple strategies, risk management, machine learning capabilities, and state-of-the-art anomaly detection.

## Features

- Multiple Trading Strategies
  - Bollinger Bands
  - Mean Reversion
  - MACD
  - Custom strategy support
- Advanced Anomaly Detection
  - Autoencoder-based detection
  - DBSCAN clustering
  - Isolation Forest
  - One-Class SVM
  - Local Outlier Factor
  - Ensemble detection with dynamic weighting
  - Market regime-aware signals
- Risk Management
  - Position sizing
  - Drawdown limits
  - Leverage control
  - Confidence-based filtering
- Machine Learning Integration
  - Ensemble models
  - Neural networks
  - GPU acceleration
- Real-time Monitoring
  - Web dashboard
  - Performance metrics
  - Trade logging
  - Anomaly visualization

## Requirements

- Python 3.8+
- NVIDIA GPU (optional, for accelerated training)
- CUDA Toolkit 11.0+ (if using GPU)
- TA-Lib (for technical analysis)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/trading-system.git
cd trading-system
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install GPU dependencies (optional):
```bash
pip install -r requirements-gpu.txt
```

## Quick Start

1. Configure your settings in `config/config.yaml`:
```yaml
trading:
  symbols: ["AAPL", "MSFT", "GOOGL"]
  initial_capital: 100000.0
  risk_per_trade: 0.02
  max_drawdown: 0.2
  max_leverage: 2.0

anomaly_detection:
  methods: ["autoencoder", "isolation_forest", "one_class_svm", "lof", "dbscan"]
  anomaly_threshold: 0.8
  confidence_threshold: 0.7
  weights:
    autoencoder: 0.3
    isolation_forest: 0.2
    one_class_svm: 0.2
    lof: 0.2
    dbscan: 0.1

data:
  start_date: "2020-01-01"
  end_date: "2023-12-31"
  interval: "1d"

model:
  type: "ensemble"  # or "neural_net", "xgboost", "random_forest"
  use_gpu: true
  batch_size: 1024
  num_workers: 4
```

2. Run the training session:
```bash
python src/blueprints/training.py
```

3. Start the web dashboard:
```bash
python src/web/app.py
```

## Project Structure

```
trading-system/
├── config/
│   └── config.yaml
├── src/
│   ├── blueprints/
│   │   └── training.py
│   ├── core/
│   │   ├── market_data/
│   │   ├── strategies/
│   │   │   ├── anomaly_detection.py
│   │   │   ├── anomaly_methods.py
│   │   │   └── ...
│   │   ├── portfolio/
│   │   └── risk/
│   ├── web/
│   │   └── app.py
│   └── utils/
├── results/
│   ├── training/
│   └── models/
├── requirements.txt
├── requirements-gpu.txt
└── setup.py
```

## Training Process

1. Data Collection
   - Fetch historical market data
   - Calculate technical indicators
   - Prepare feature sets

2. Anomaly Detection Training
   - Train autoencoder
   - Train DBSCAN
   - Train Isolation Forest
   - Train One-Class SVM
   - Train Local Outlier Factor
   - Optimize ensemble weights

3. Strategy Testing
   - Run backtests on multiple strategies
   - Optimize strategy parameters
   - Evaluate performance metrics

4. Model Training
   - Prepare training data
   - Train selected model type
   - Validate and save results

## Web Dashboard

Access the dashboard at `http://localhost:8000` to:
- Monitor real-time trading
- View portfolio performance
- Analyze strategy results
- Adjust trading parameters
- Visualize anomaly detection
- Monitor market regimes

## GPU Acceleration

The system supports GPU acceleration for:
- XGBoost training
- Neural network training
- Data processing
- Anomaly detection (autoencoder)

To enable GPU support:
1. Install NVIDIA drivers
2. Install CUDA Toolkit
3. Set `use_gpu: true` in config

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue in the GitHub repository.