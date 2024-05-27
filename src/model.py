import logging
import joblib
import pandas as pd
from utils import execute_trade
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def load_model(filepath='src/optimized_pump_dump_model.pkl'):
    return joblib.load(filepath)

def train_ensemble_model(X_train, y_train):
    best_models = load_model()
    ensemble_model = VotingClassifier(estimators=[
        ('rf', best_models['RandomForest']),
        ('gb', best_models['GradientBoosting']),
        ('svm', best_models['SVM'])
    ], voting='soft')
    ensemble_model = MultiOutputClassifier(ensemble_model)
    ensemble_model.fit(X_train, y_train)
    return ensemble_model

def evaluate_ensemble_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Ensemble Model")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

def save_model(model, file_path='optimized_pump_dump_ensemble_model.pkl'):
    joblib.dump(model, file_path)

def train_log_reg_model(X_train, y_train):
    log_reg = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', random_state=42)
    log_reg = MultiOutputClassifier(log_reg)
    log_reg.fit(X_train, y_train)
    return log_reg

def evaluate_log_reg_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Logistic Regression with L2 Regularization")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

def preprocess_data(data):
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

    data = data.dropna()
    return data

def predict_anomaly(model, historical_data, TRADE_AMOUNT):
    try:
        historical_data = preprocess_data(historical_data)
        features = ['price_change', 'volume_change', 'ma_10', 'ma_50', 'ma_200', 'ma_diff', 'std_10', 'std_50', 'momentum', 'volatility', 'rsi', 'macd', 'bb_high', 'bb_low']
        X = historical_data[features]

        predictions = model.predict(X)

        if predictions[-1] == 1:  # Assuming 1 indicates an anomaly
            logging.info(f"Anomaly detected: Executing trade for amount {TRADE_AMOUNT}")
            execute_trade('buy', 5)
        else:
            logging.info("No anomaly detected")

        return predictions
    except Exception as e:
        logging.error(f"Error in predict_anomaly: {e}")
        return None

data = pd.read_csv('path_to_your_data.csv')
features = ['price_change', 'volume_change', 'ma_10', 'ma_50', 'ma_200', 'ma_diff', 'std_10', 'std_50', 'momentum', 'volatility', 'rsi', 'macd']

data = preprocess_data(data)
X = data[features]
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ensemble_model = train_ensemble_model(X_train, y_train)
evaluate_ensemble_model(ensemble_model, X_test, y_test)
save_model(ensemble_model)

log_reg = train_log_reg_model(X_train, y_train)
evaluate_log_reg_model(log_reg, X_test, y_test)
save_model(log_reg, 'optimized_pump_dump_log_reg_model.pkl')
