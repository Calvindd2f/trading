import logging
import joblib
import pandas as pd
from utils import execute_trade
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from sklearn.model_selection import train_test_split

def load_model(filepath='src/optimized_pump_dump_model.pkl'):
    return joblib.load(filepath, mmap_mode='r')

# Load preprocessed data
def load_best_models(file_path='src/optimized_pump_dump_model.pkl'):
    return joblib.load(file_path)

best_models = load_best_models()

# Create an ensemble of the best models
ensemble_model = VotingClassifier(estimators=[
    ('rf', best_models['RandomForest']),
    ('gb', best_models['GradientBoosting']),
    ('svm', best_models['SVM'])
], voting='soft')

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

def train_ensemble_model(model, X_train, y_train):
    model.fit(X_train, y_train, n_jobs=-1)
    return model

ensemble_model = train_ensemble_model(ensemble_model, X_train, y_train)

def evaluate_ensemble_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Ensemble Model")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

evaluate_ensemble_model(ensemble_model, X_test, y_test)

# Save the ensemble model
def save_model(model, file_path='optimized_pump_dump_ensemble_model.pkl'):
    joblib.dump(model, file_path)

save_model(ensemble_model)

# Training and evaluation of the Logistic Regression model with L2 regularization
def train_log_reg_model(X_train, y_train):
    log_reg = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', random_state=42)
    log_reg.fit(X_train, y_train)
    return log_reg

def evaluate_log_reg_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Logistic Regression with L2 Regularization")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

log_reg = train_log_reg_model(X_train, y_train)
evaluate_log_reg_model(log_reg, X_test, y_test)

# Save the regularized model
save_model(log_reg, 'optimized_pump_dump_log_reg_model.pkl')

def preprocess_data(data):
    # Feature engineering
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

    # Drop NaN values after adding features
    data = data.dropna()
    return data

# Assuming `data` is your input DataFrame
data = pd.read_csv('path_to_your_data.csv')  # Load your data
features = ['price_change', 'volume_change', 'ma_10', 'ma_50', 'ma_200', 'ma_diff', 'std_10', 'std_50', 'momentum', 'volatility', 'rsi', 'macd']


data = preprocess_data(data)
X = data[features]
y = data['label']  # Assuming there is a 'label' column for supervised learning

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def predict_anomaly(model, historical_data, TRADE_AMOUNT):
    try:
        # Feature engineering
        historical_data['price_change'] = historical_data['price'].pct_change()
        historical_data['volume_change'] = historical_data['volume'].pct_change()
        historical_data['ma_10'] = historical_data['price'].rolling(window=10).mean()
        historical_data['ma_50'] = historical_data['price'].rolling(window=50).mean()
        historical_data['ma_200'] = historical_data['price'].rolling(window=200).mean()
        historical_data['ma_diff'] = historical_data['ma_10'] - historical_data['ma_50']
        historical_data['std_10'] = historical_data['price'].rolling(window=10).std()
        historical_data['std_50'] = historical_data['price'].rolling(window=50).std()
        historical_data['momentum'] = historical_data['price'] - historical_data['price'].shift(4)
        historical_data['volatility'] = historical_data['price'].rolling(window=10).std()
        
        rsi = RSIIndicator(close=historical_data['price'])
        historical_data['rsi'] = rsi.rsi()
        
        macd = MACD(close=historical_data['price'])
        historical_data['macd'] = macd.macd()
        
        bollinger = BollingerBands(close=historical_data['price'])
        historical_data['bb_high'] = bollinger.bollinger_hband()
        historical_data['bb_low'] = bollinger.bollinger_lband()
        
        features = ['price_change', 'volume_change', 'ma_10', 'ma_50', 'ma_200', 'ma_diff', 'std_10', 'std_50', 'momentum', 'volatility', 'rsi', 'macd', 'bb_high', 'bb_low']
        historical_data = historical_data.dropna()
        
        X = historical_data[features]
        
        # Model predictions
        predictions = model.predict(X)
        
        # Model evaluation (optional, for monitoring and debugging)
        y_true = historical_data['label']  # Assuming there is a 'label' column for actual anomalies
        accuracy = accuracy_score(y_true, predictions)
        precision = precision_score(y_true, predictions)
        recall = recall_score(y_true, predictions)
        
        logging.info(f"Model Accuracy: {accuracy:.2f}")
        logging.info(f"Model Precision: {precision:.2f}")
        logging.info(f"Model Recall: {recall:.2f}")
        
        # Further processing based on predictions
        if predictions[-1] == 1:  # Assuming 1 indicates an anomaly
            logging.info(f"Anomaly detected: Executing trade for amount {TRADE_AMOUNT}")
            execute_trade( 'buy', 5)
        else:
            logging.info("No anomaly detected")
        
        return predictions
    except Exception as e:
        logging.error(f"Error in predict_anomaly: {e}")
        return None
