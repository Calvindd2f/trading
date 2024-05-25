import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import joblib
import logging
import random

def load_data(filepath):
    """
    Load data from a CSV file.

    Parameters:
        filepath (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded data as a DataFrame.
    """
    return pd.read_csv(filepath)

def preprocess_data(data):
    data['price_change'] = data['price'].pct_change().values
    data['volume_change'] = data['volume'].pct_change().values
    data['ma_10'] = data['price'].rolling(window=10).mean().values
    data['ma_50'] = data['price'].rolling(window=50).mean().values
    data['ma_200'] = data['price'].rolling(window=200).mean().values
    data['ma_diff'] = data['ma_10'] - data['ma_50']
    data['std_10'] = data['price'].rolling(window=10).std().values
    data['std_50'] = data['price'].rolling(window=50).std().values
    data['momentum'] = data['price'] - data['price'].shift(4)
    data['volatility'] = data['price'].rolling(window=20).std() / data['price'].rolling(window=20).mean()
    data['rsi'] = calculate_rsi(data['price'].values)
    data['macd'] = calculate_macd(data['price'].values)
    data.dropna(inplace=True)
    return data

def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, slow=26, fast=12, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd - signal_line

def train_model(data, features):
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
    for _ in range(n_passes):
        selected_cryptos = select_random_cryptos(data, n_cryptos)
        logging.info(f"Selected cryptos for this pass: {selected_cryptos}")
        subset = data[data['crypto'].isin(selected_cryptos)]
        processed_data = preprocess_data(subset)
        best_models = train_model(processed_data, features)
        best_model = best_models['GradientBoosting']  # or select based on performance
        results.append(evaluate_model(best_model, processed_data, features))

    return aggregate_results(results)

def evaluate_model(model, data, features):
    X = data[features]
    y = data['label']
    y_pred = model.predict(X)
    gain_loss = calculate_gain_loss(y, y_pred)
    return gain_loss

def calculate_gain_loss(y_true, y_pred):
    # Simplified example to calculate gain/loss
    return np.sum((y_pred == y_true) * 1.0) - np.sum((y_pred != y_true) * 1.0)

def aggregate_results(results):
    return np.mean(results)

if __name__ == "__main__":
    data = load_data('data/historical_data.csv')
    features = ['price_change', 'volume_change', 'ma_10', 'ma_50', 'ma_200', 'ma_diff', 'std_10', 'std_50', 'momentum', 'volatility', 'rsi', 'macd']
    final_result = three_pass_training(data, features)
    logging.info(f"Final result after three passes: {final_result}")
    if final_result > 0:
        best_model = train_model(data, features)['GradientBoosting']
        save_model(best_model, 'src/optimized_pump_dump_model.pkl')
    else:
        logging.warning("Training failed to achieve positive gain/loss. Model not updated.")
