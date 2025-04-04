import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# API Configuration
API_CONFIG = {
    "yfinance": {
        "enabled": True,
        "cache_duration": 3600,  # 1 hour
    },
    "alpha_vantage": {
        "enabled": True,
        "api_key": os.getenv("ALPHA_VANTAGE_API_KEY"),
        "rate_limit": 5,  # calls per minute
    },
    "finnhub": {
        "enabled": True,
        "api_key": os.getenv("FINNHUB_API_KEY"),
        "rate_limit": 60,  # calls per minute
    }
}

# Database Configuration
DATABASE_CONFIG = {
    "type": "sqlite",  # or "postgresql", "mongodb"
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "name": os.getenv("DB_NAME", "trading_db"),
    "user": os.getenv("DB_USER", ""),
    "password": os.getenv("DB_PASSWORD", ""),
}

# Trading Configuration
TRADING_CONFIG = {
    "default_timeframe": "1d",
    "supported_timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"],
    "max_position_size": 0.1,  # 10% of portfolio
    "stop_loss_pct": 0.02,    # 2% stop loss
    "take_profit_pct": 0.04,  # 4% take profit
}

# Model Configuration
MODEL_CONFIG = {
    "retraining_interval": 24,  # hours
    "lookback_window": 30,      # days
    "prediction_horizon": 1,    # days
    "min_confidence": 0.7,      # minimum confidence for trades
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": BASE_DIR / "logs" / "trading.log",
}

# Create necessary directories
for directory in [DATA_DIR, MODELS_DIR, LOGGING_CONFIG["file"].parent]:
    directory.mkdir(parents=True, exist_ok=True) 