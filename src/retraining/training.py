import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import joblib
import logging
import random
from ta import add_all_ta_features as talib
from typing import List, Optional
from numba import jit

# Configure logging
logging.basicConfig(level=logging.INFO)

def load_data(filepath):
    return pd.read_csv(filepath)

def preprocess_data(data, future_period=1):
    assert isinstance(data, pd.DataFrame), "data must be a DataFrame"
    assert 'price' in data.columns and 'volume' in data.columns, "data must contain columns 'price' and 'volume'"
    assert isinstance(future_period, int) and future_period > 0, "future_period must be a positive integer"

    data['price_change'] = data['price'].pct_change().values
    data['volume_change'] = data['volume'].pct_change().values

    # Calculate the moving averages
    data['ma_10'] = data['price'].rolling(window=10).mean().values
    data['ma_50'] = data['price'].rolling(window=50).mean().values
    data['ma_200'] = data['price'].rolling(window=200).mean().values

    # Calculate the difference between the short and long moving averages
    data['ma_diff'] = data['ma_10'] - data['ma_50']

    # Calculate the standard deviation over different time periods
    data['std_10'] = data['price'].rolling(window=10).std().values
    data['std_50'] = data['price'].rolling(window=50).std().values

    # Calculate the momentum
    data['momentum'] = data['price'].rolling(window=4).mean().values

    # Calculate the volatility
    data['volatility'] = data['price'].rolling(window=20).std().values / data['price'].rolling(window=20).mean().values

    # Calculate the Relative Strength Index
    data['rsi'] = talib.RSI(data['price'], timeperiod=14)

    # Calculate the Moving Average Convergence Divergence
    data['macd'], _, _ = talib.MACD(data['price'], fastperiod=12, slowperiod=26, signalperiod=9)

    # Create the label column
    data['label'] = create_labels(data['price'], future_period)

    # Drop any rows with missing values
    data.dropna(inplace=True)

    print("Columns after preprocessing: ", data.columns.tolist())  # Debugging statement
    return data

def create_labels(price_series: pd.Series, future_period: int = 1) -> pd.Series:
    """
    Create a series of labels from the price series. The labels are 1 if the future price is higher than the current price, and 0 otherwise.

    Parameters
    ----------
    price_series : pandas.Series
        The price series to use for calculating the labels.
    future_period : int, optional
        The period in the future to look at the price. Defaults to 1.

    Returns
    -------
    pandas.Series
        A series of labels. The labels are 1 if the future price is higher than the current price, and 0 otherwise.
    """
    future_price = price_series.shift(-future_period)
    label = (future_price > price_series).astype(int)
    return label

def train_model(data, features):
    """
    Train and evaluate several models using the provided DataFrame and features.

    Parameters
    ----------
    data : pandas.DataFrame
        The DataFrame containing the data to be used for training.
    features : list
        The names of the features used for training.

    Returns
    -------
    dict
        A dictionary containing the best models for each model type.
    """
    features_data = data[features]
    y = data['label']

    train_features, test_features, y_train, _ = train_test_split(features_data, y, test_size=0.2, random_state=42)

    models = {
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42)
    }

    param_grids = {
        'GradientBoosting': [
            {'n_estimators': [100], 'learning_rate': [0.1], 'max_depth': [3], 'subsample': [0.8]},
            {'n_estimators': [200], 'learning_rate': [0.1], 'max_depth': [5], 'subsample': [0.8]},
            {'n_estimators': [300], 'learning_rate': [0.1], 'max_depth': [7], 'subsample': [1.0]}
        ],
        'SVM': [
            {'C': [0.1], 'gamma': [0.001]},
            {'C': [1], 'gamma': [0.01]},
            {'C': [10], 'gamma': [0.1]}
        ]
    }

    best_models = {}

    for model_name in models:
        grid_search = GridSearchCV(estimator=models[model_name], param_grid=param_grids[model_name], cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(train_features, y_train)
        best_models[model_name] = grid_search.best_estimator_
        logging.info("Best parameters for %s: %s", model_name, grid_search.best_params_)

    return best_models

def save_model(model, filepath):
    joblib.dump(model, filepath)
    logging.info("Model saved to %s", filepath)

def select_random_cryptos(data, n=3):
    cryptos = data['crypto'].unique()
    return random.sample(list(cryptos), n)

def three_pass_training(data: pd.DataFrame, features: List[str], n_passes: int = 3, n_cryptos: int = 3) -> Optional[GradientBoostingClassifier]:
    """
    Perform three passes of training on the given data.

    Args:
        data (pandas.DataFrame): The DataFrame containing the data to be used for training.
        features (list): The names of the features used for training.
        n_passes (int): The number of passes to perform. Defaults to 3.
        n_cryptos (int): The number of cryptos to select for each pass. Defaults to 3.

    Returns:
        Optional[GradientBoostingClassifier]: The best model after three passes, or None if the final result is negative.
    """
    results = []
    best_model = None
    cryptos = data['crypto'].unique()
    random_cryptos = random.sample(list(cryptos), n_passes * n_cryptos)
    for i in range(n_passes):
        selected_cryptos = random_cryptos[i * n_cryptos:(i + 1) * n_cryptos]
        logging.info("Selected cryptos for this pass: %s", selected_cryptos)
        subset = data[data['crypto'].isin(selected_cryptos)]
        processed_data = preprocess_data(subset)
        best_models = train_model(processed_data, features)
        best_model = best_models['GradientBoosting']  # or select based on performance
        results.append(evaluate_model(best_model, processed_data, features))

    final_result = aggregate_results(results)
    logging.info("Final result after three passes: %s", final_result)
    return best_model if final_result > 0 else None

def evaluate_model(model: GradientBoostingClassifier, data: pd.DataFrame, features: List[str]) -> float:
    """
    Evaluate a trained machine learning model on a given dataset.

    Args:
        model (GradientBoostingClassifier): The trained machine learning model.
        data (pandas.DataFrame): The dataset to evaluate the model on.
        features (list): The list of features to use for prediction.

    Returns:
        float: The mean gain or loss of the model's predictions compared to the true labels.
    """
    if model is None:
        raise ValueError("Model cannot be None")
    if data is None:
        raise ValueError("Data cannot be None")
    if features is None:
        raise ValueError("Features cannot be None")

    features_data = data[features].values
    y = data['label'].values

    if features_data is None or y is None or len(features_data) != len(y):
        raise ValueError("Features and labels must have the same length")

    y_pred = model.predict(features_data)
    if y_pred is None:
        raise ValueError("Model did not return a prediction")

    return np.mean((y == y_pred) * 1.0) - np.mean((y != y_pred) * 1.0)

def calculate_gain_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the mean gain or loss of the model's predictions compared to the true labels.

    Args:
        y_true (numpy.ndarray): The true labels.
        y_pred (numpy.ndarray): The predicted labels.

    Returns:
        float: The mean gain or loss of the model's predictions compared to the true labels.
    """
    return np.sum((y_pred == y_true) * 1.0) - np.sum((y_pred != y_true) * 1.0)

@jit
def aggregate_results(results: List[float]) -> float:
    """
    Calculate the mean of a list of results.

    Args:
        results (list): A list of results to calculate the mean of.

    Returns:
        float: The mean of the results.
    """
    return np.mean(results)

if __name__ == "__main__":
    data = load_data('data/historical_data.csv')
    features = ['price_change', 'volume_change', 'ma_10', 'ma_50', 'ma_200', 'ma_diff', 'std_10', 'std_50', 'momentum', 'volatility', 'rsi', 'macd']
    best_model = three_pass_training(data, features)
    if best_model:
        save_model(best_model, 'src/optimized_pump_dump_model.pkl')
    else:
        logging.warning("Training failed to achieve positive gain/loss. Model not updated.")
