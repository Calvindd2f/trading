import pandas as pd
import joblib
from datetime import datetime, timedelta


def preprocess_data(df):
    df['price_change'] = df['price'].pct_change()
    df['volume_change'] = df['volume'].pct_change()
    df['ma_10'] = df['price'].rolling(window=10).mean()
    df['ma_50'] = df['price'].rolling(window=50).mean()
    df['ma_200'] = df['price'].rolling(window=200).mean()
    df['ma_diff'] = df['ma_10'] - df['ma_50']
    df['std_10'] = df['price'].rolling(window=10).std()
    df['std_50'] = df['price'].rolling(window=50).std()
    df['momentum'] = df['price'] - df['price'].shift(4)
    df['volatility'] = df['price'].rolling(window=20).std() / df['price'].rolling(window=20).mean()
    df['label'] = 0  # Default label for normal behavior
    df.loc[df['price_change'] >= BUY_THRESHOLD, 'label'] = 1  # Label for pump
    df.loc[df['price_change'] <= SELL_THRESHOLD, 'label'] = -1  # Label for dump
    df.dropna(inplace=True)
    return df

# Load historical data
symbols = ["BTCUSD", "ETHUSD", "LTCUSD"]
all_data = pd.concat([preprocess_data(fetch_historical_data(symbol)) for symbol in symbols])

# Save preprocessed data
all_data.to_csv('historical_data.csv', index=False)

# Load the trained model
model = joblib.load('pump_dump_model.pkl')

# Load historical data from the database
def fetch_historical_data_from_db():
    conn = sqlite3.connect('trading_bot.db')
    df = pd.read_sql_query("SELECT * FROM historical_data", conn)
    conn.close()
    return df

# Backtest the trading bot using historical data
def backtest_trading_bot(historical_data):
    global total_loss, trade_count, backoff_duration
    total_loss = 0
    trade_count = 0
    backoff_duration = INITIAL_BACKOFF
    
    for index, row in historical_data.iterrows():
        timestamp = row['time']
        price = row['price']
        volume = row['volume']
        process_historical_data(timestamp, price, volume)
    
    print(f"Total Trades: {trade_count}")
    print(f"Total Loss: {total_loss}")


# Load preprocessed data
data = pd.read_csv('historical_data.csv')

# Define features and labels
features = ['price_change', 'volume_change', 'ma_10', 'ma_50', 'ma_200', 'ma_diff', 'std_10', 'std_50', 'momentum', 'volatility']
X = data[features]
y = data['label']

# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluate different models
models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42)
}

for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Model: {model_name}")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")

# Example for RandomForest
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                           param_grid=param_grid,
                           cv=3,
                           n_jobs=-1,
                           verbose=2)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
best_model = grid_search.best_estimator_

# Evaluate the best model
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")


