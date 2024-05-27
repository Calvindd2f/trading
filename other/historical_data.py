import pandas as pd
import requests
import logging
from joblib import dump

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set API base URL, buy, and sell thresholds as global constants
API_BASE_URL = "https://api.example.com"
BUY_THRESHOLD = 0.05
SELL_THRESHOLD = -0.0
