import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands
from pathlib import Path

# Replace this with the actual path to your data file
DATA_FILE_PATH = Path('data/historical_data.csv')

def import_feature_engineering_module():
    try:
        module_path = Path('feature_engineering.py')
        if module_path.exists():
            import feature_engineering  # Ensure this import matches your file structure
            return feature_engineering
        else:
            raise ModuleNotFoundError(f"File '{module_path}' not found")
    except ModuleNotFoundError as e:
        print(f"Error: {e}")
        return None

feature_engineering = import_feature_engineering_module()
if feature_engineering:
    # Load your data
    data = pd.read_csv(DATA_FILE_PATH)

    # Apply feature engineering
    if 'preprocess_data' in dir(feature_engineering):
        data = feature_engineering.preprocess_data(data)
    else:
        print("Warning: 'preprocess_data' function not found in feature_engineering module")

    # Define features
    FEATURES = ['price_change', 'volume_change', 'ma_10', 'ma_50', 'ma_200', 'ma_diff', 'std_10', 'std_50', 'momentum', 'volatility', 'rsi', 'macd']

    # Ensure all features are present
    for feature in FEATURES:
        if feature not in data.columns:
            raise KeyError(f"Feature {feature} not found in data columns")

    # Select features for training
    X = data[FEATURES]
    y = data['label']  # Assuming there is a 'label' column for supervised learning

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    print("Error: Unable to import feature_engineering module")
