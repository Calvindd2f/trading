import joblib
import pandas as pd
from utils import execute_trade
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import VotingClassifier,
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression

# Load preprocessed data
best_models = joblib.load('optimized_pump_dump_model.pkl')

# Create an ensemble of the best models
ensemble_model = VotingClassifier(estimators=[
    ('rf', best_models['RandomForest']),
    ('gb', best_models['GradientBoosting']),
    ('svm', best_models['SVM'])
], voting='soft')

ensemble_model.fit(X_train, y_train)
y_pred = ensemble_model.predict(X_test)
print("Ensemble Model")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Save the ensemble model
joblib.dump(ensemble_model, 'optimized_pump_dump_ensemble_model.pkl')

model = joblib.load('optimized_pump_dump_model.pkl')
# Cross-validation
cv_scores = cross_val_score(best_model, X, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Average CV score: {cv_scores.mean()}")

def load_model(model_path='optimized_pump_dump_model.pkl'):
    return joblib.load(model_path)

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
    data['rsi'] = calculate_rsi(data['price'])
    data['macd'] = calculate_macd(data['price'])
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

def predict_anomaly():
    global historical_data

    latest_data = preprocess_data(historical_data.iloc[-1:])
    features = ['price_change', 'volume_change', 'ma_10', 'ma_50', 'ma_200', 'ma_diff', 'std_10', 'std_50', 'momentum', 'volatility', 'rsi', 'macd']
    X_latest = latest_data[features]

    prediction = model.predict(X_latest)[0]

    if prediction == 1:
        execute_trade("buy", TRADE_AMOUNT)
    elif prediction == -1:
        execute_trade("sell", TRADE_AMOUNT)

# Example of logistic regression with L2 regularization
log_reg = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', random_state=42)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
print("Logistic Regression with L2 Regularization")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Save the regularized model
joblib.dump(log_reg, 'optimized_pump_dump_log_reg_model.pkl')