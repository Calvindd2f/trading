import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands
from feature_engineering import preprocess_data  # Ensure this import matches your file structure

# Load your data
data = pd.read_csv('data/historical_data.csv')  # Replace with your actual data file path

# Apply feature engineering
data = preprocess_data(data)

# Define features
features = ['price_change', 'volume_change', 'ma_10', 'ma_50', 'ma_200', 'ma_diff', 'std_10', 'std_50', 'momentum', 'volatility', 'rsi', 'macd']

# Ensure all features are present
for feature in features:
    if feature not in data.columns:
        raise KeyError(f"Feature {feature} not found in data columns")

# Select features for training
X = data[features]
y = data['label']  # Assuming there is a 'label' column for supervised learning

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)