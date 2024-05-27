from sklearn.metrics import accuracy_score

def evaluate_model(model: LinearRegression, X_test: pd.DataFrame, y_test: pd.DataFrame) -> float:
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy