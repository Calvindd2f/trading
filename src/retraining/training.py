import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import joblib
import logging
import random
from ta import add_all_ta_features as talib
from model_tuning import y_train, X_train, X_test, y_test

# Configure logging
logging.basicConfig(level=logging.INFO)

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(data, future_period=1):
    data['price_change'] = data['price'].pct_change().values
    data['volume_change'] = data['volume'].pct_change().values
    data['ma_10'] = talib.SMA(data['price'], timeperiod=10)
    data['ma_50'] = talib.SMA(data['price'], timeperiod=50)
    data['ma_200'] = talib.SMA(data['price'], timeperiod=200)
    data['ma_diff'] = data['ma_10'] - data['ma_50']
    data['std_10'] = data['price'].rolling(window=10).std().values
    data['std_50'] = data['price'].rolling(window=50).std().values
    data['momentum'] = talib.MOM(data['price'], timeperiod=4)
    data['volatility'] = data['price'].rolling(window=20).std() / data['price'].rolling(window=20).mean()
    data['rsi'] = talib.RSI(data['price'], timeperiod=14)
    data['macd'], macdsignal, macdhist = talib.MACD(data['price'], fastperiod=12, slowperiod=26, signalperiod=9)
    data['label'] = create_labels(data['price'], future_period)
    data.dropna(inplace=True)
    print("Columns after preprocessing: ", data.columns.tolist())  # Debugging statement
    return data

def create_labels(price_series, future_period=1):
    future_price = price_series.shift(-future_period)
    label = (future_price > price_series).astype(int)
    return label

def train_model(data, features):
    print("Features available in DataFrame: ", data.columns.tolist())  # Debugging statement
    X = data[features]
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42)
    }

    param_grids = {
        'GradientBoosting': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'gamma': [0.001, 0.01, 0.1]
        }
    }

    best_models = {}

    for model_name in models:
        grid_search = GridSearchCV(estimator=models[model_name], param_grid=param_grids[model_name], cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        best_models[model_name] = grid_search.best_estimator_
        logging.info(f"Best parameters for {model_name}: {grid_search.best_params_}")

    return best_models

def save_model(model, filepath):
    joblib.dump(model, filepath)
    logging.info(f"Model saved to {filepath}")

def select_random_cryptos(data, n=3):
    cryptos = data['crypto'].unique()
    return random.sample(list(cryptos), n)

def three_pass_training(data, features, n_passes=3, n_cryptos=3):
    results = []
    best_model = None
    for _ in range(n_passes):
        selected_cryptos = select_random_cryptos(data, n_cryptos)
        logging.info(f"Selected cryptos for this pass: {selected_cryptos}")
        subset = data[data['crypto'].isin(selected_cryptos)]
        processed_data = preprocess_data(subset)
        best_models = train_model(processed_data, features)
        best_model = best_models['GradientBoosting']  # or select based on performance
        results.append(evaluate_model(best_model, processed_data, features))

    final_result = aggregate_results(results)
    logging.info(f"Final result after three passes: {final_result}")
    return best_model if final_result > 0 else None

def evaluate_model(model, data, features):
    X = data[features]
    y = data['label']
    y_pred = model.predict(X)
    gain_loss = calculate_gain_loss(y, y_pred)
    return gain_loss

def calculate_gain_loss(y_true, y_pred):
    return np.sum((y_pred == y_true) * 1.0) - np.sum((y_pred != y_true) * 1.0)

def aggregate_results(results):
    return np.mean(results)

if __name__ == "__main__":
    data = load_data('data/historical_data.csv')
    features = ['price_change', 'volume_change', 'ma_10', 'ma_50', 'ma_200', 'ma_diff', 'std_10', 'std_50', 'momentum', 'volatility', 'rsi', 'macd']
    best_model = three_pass_training(data, features)
    if best_model:
        save_model(best_model, 'src/optimized_pump_dump_model.pkl')
    else:
        logging.warning("Training failed to achieve positive gain/loss. Model not updated.")
