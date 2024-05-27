import logging
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from typing import Dict, List, Tuple

def optimize_algorithm(data: pd.DataFrame) -> Tuple[Dict[str, int], float]:
    """Optimize algorithm parameters using GridSearchCV.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing features and target

    Returns
    -------
    Tuple[Dict[str, int], float]
        Best parameters and best score
    """
    logging.info("Optimizing algorithm parameters...")
    parameters: Dict[str, List[int]] = {
        'max_depth': [3, 5, 10],
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1],
    }
    model = GradientBoostingClassifier()
    grid_search = GridSearchCV(model, parameters, cv=5, scoring='f1_macro')
    grid_search.fit(data.drop('label', axis=1), data['label'])
    logging.info(f"Best parameters: {grid_search.best_params_}")
    logging.info(f"Best score: {grid_search.best_score_}")
    return grid_search.best_params_, grid_search.best_score_

def load_data(file_path):
    # Load data from a CSV file
    logging.info(f"Loading data from {file_path}...")
    data = pd.read_csv(file_path)
    return data

def save_results(data, file_path):
    # Save results to a CSV file
    logging.info(f"Saving results to {file_path}...")
    data.to_csv(file_path, index=False)

if __name__ == "__main__":
    file_path = 'data/historical_data.csv'
    data = load_data(file_path)
    assess_cryptos(data)
    optimize_algorithm(data)
    output_file_path = 'data/results.csv'
    save_results(data, output_file_path)
