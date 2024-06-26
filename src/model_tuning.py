import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load preprocessed data
def load_data(filepath):
    return pd.read_csv(filepath)

# Define features and labels
def get_features_and_labels(data):
    features = [
        'price_change', 'volume_change', 'ma_10', 'ma_50', 'ma_200', 
        'ma_diff', 'std_10', 'std_50', 'momentum', 'volatility', 'rsi', 'macd'
    ]
    X = data[features]
    y = data['label']
    return X, y

# Model selection and hyperparameter tuning
def tune_models(X_train, y_train):
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42)
    }

    param_grids = {
        'RandomForest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'GradientBoosting': {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'gamma': [0.001, 0.01, 0.1],
        }
    }

    best_models = {}

    for model_name in models:
        logging.info(f"Starting GridSearch for {model_name}")
        grid_search = GridSearchCV(estimator=models[model_name], param_grid=param_grids[model_name], cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        best_models[model_name] = grid_search.best_estimator_
        logging.info(f"Best parameters for {model_name}: {grid_search.best_params_}")

    return best_models

# Evaluate best models
def evaluate_models(models, X_test, y_test):
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        logging.info(f"Model: {model_name}")
        logging.info(f"\n{classification_report(y_test, y_pred)}")
        logging.info(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")

# Save the best model
def save_best_model(model, filepath):
    joblib.dump(model, filepath)
    logging.info(f"Model saved to {filepath}")

def main():
    # Load data
    data = load_data('data/historical_data.csv')

    # Get features and labels
    X, y = get_features_and_labels(data)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tune models
    best_models = tune_models(X_train, y_train)

    # Evaluate models
    evaluate_models(best_models, X_test, y_test)

    # Save the best model (adjust based on your evaluation results)
    best_model = best_models['GradientBoosting']  # Replace with the model that performs the best
    save_best_model(best_model, 'src/optimized_pump_dump_model.pkl')

if __name__ == "__main__":
    main()
