import logging
import joblib
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from typing import List

# Utility function to execute trades (dummy function for simulation)
def execute_trade(side: str, amount: float):
    logging.info(f"Executing trade: {side} {amount}")

# Load the saved model from file
def load_model(filepath='src/optimized_pump_dump_model.pkl'):
    return joblib.load(filepath, mmap_mode='r')

# Load the best models and create an ensemble
def load_best_models(file_path='src/optimized_pump_dump_model.pkl'):
    return joblib.load(file_path)

best_models = load_best_models()

ensemble_model = VotingClassifier(estimators=[
    ('rf', best_models['RandomForest']),
    ('gb', best_models['GradientBoosting']),
    ('svm', best_models['SVM'])
], voting='soft')

# Function to preprocess the data
def preprocess_data(data: pd.DataFrame, future_period=1) -> pd.DataFrame:
    data['price_change'] = data['price'].pct_change()
    data['volume_change'] = data['volume'].pct_change()
    data['ma_10'] = data['price'].rolling(window=10).mean()
    data['ma_50'] = data['price'].rolling(window=50).mean()
    data['ma_200'] = data['price'].rolling(window=200).mean()
    data['ma_diff'] = data['ma_10'] - data['ma_50']
    data['std_10'] = data['price'].rolling(window=10).std()
    data['std_50'] = data['price'].rolling(window=50).std()
    data['momentum'] = data['price'] - data['price'].shift(4)
    data['volatility'] = data['price'].rolling(window=10).std()

    rsi = RSIIndicator(close=data['price'])
    data['rsi'] = rsi.rsi()

    macd = MACD(close=data['price'])
    data['macd'] = macd.macd()

    bollinger = BollingerBands(close=data['price'])
    data['bb_high'] = bollinger.bollinger_hband()
    data['bb_low'] = bollinger.bollinger_lband()

    data['label'] = create_labels(data['price'], future_period)
    data.dropna(inplace=True)
    
    # Drop NaN values after adding features
    data = data.dropna()
    return data

def create_labels(price_series: pd.Series, future_period: int = 1) -> pd.Series:
    future_price = price_series.shift(-future_period)
    label = (future_price > price_series).astype(int)
    return label

# Function to train the ensemble model
def train_ensemble_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

# Function to evaluate the ensemble model
def evaluate_ensemble_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Ensemble Model")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Function to save the model to a file
def save_model(model, file_path='optimized_pump_dump_ensemble_model.pkl'):
    joblib.dump(model, file_path)
    logging.info(f"Model saved to {file_path}")

# Function to train and evaluate Logistic Regression model
def train_log_reg_model(X_train, y_train):
    log_reg = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', random_state=42)
    log_reg.fit(X_train, y_train)
    return log_reg

def evaluate_log_reg_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Logistic Regression with L2 Regularization")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Function to predict anomalies using the model
def predict_anomaly(model, historical_data: pd.DataFrame, trade_amount: float):
    try:
        historical_data = preprocess_data(historical_data)
        features = ['price_change', 'volume_change', 'ma_10', 'ma_50', 'ma_200', 'ma_diff', 'std_10', 'std_50', 'momentum', 'volatility', 'rsi', 'macd', 'bb_high', 'bb_low']
        X = historical_data[features]

        predictions = model.predict(X)
        y_true = historical_data['label']
        accuracy = accuracy_score(y_true, predictions)
        precision = precision_score(y_true, predictions)
        recall = recall_score(y_true, predictions)

        logging.info(f"Model Accuracy: {accuracy:.2f}")
        logging.info(f"Model Precision: {precision:.2f}")
        logging.info(f"Model Recall: {recall:.2f}")

        if predictions[-1] == 1:
            logging.info(f"Anomaly detected: Executing trade for amount {trade_amount}")
            execute_trade('buy', trade_amount)
        else:
            logging.info("No anomaly detected")

        return predictions
    except Exception as e:
        logging.error(f"Error in predict_anomaly: {e}")
        return None

# Main function to orchestrate training and evaluation
def main():
    # Load and preprocess data
    data = pd.read_csv('data/historical_data.csv')
    data = preprocess_data(data)
    features = ['price_change', 'volume_change', 'ma_10', 'ma_50', 'ma_200', 'ma_diff', 'std_10', 'std_50', 'momentum', 'volatility', 'rsi', 'macd']
    X = data[features]
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate ensemble model
    ensemble_model = train_ensemble_model(ensemble_model, X_train, y_train)
    evaluate_ensemble_model(ensemble_model, X_test, y_test)
    save_model(ensemble_model, 'optimized_pump_dump_ensemble_model.pkl')

    # Train and evaluate Logistic Regression model
    log_reg = train_log_reg_model(X_train, y_train)
    evaluate_log_reg_model(log_reg, X_test, y_test)
    save_model(log_reg, 'optimized_pump_dump_log_reg_model.pkl')

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()