import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, TimeSeriesSplit
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import joblib
import logging
import random
from ta import add_all_ta_features as talib
from typing import List, Optional, Dict, Tuple
from numba import jit, prange
import optuna
from optuna.samplers import TPESampler
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import gc
import psutil
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)

class AdvancedModelTrainer:
    """Enhanced model training class with advanced optimization and parallel processing."""
    
    def __init__(self, use_ray: bool = True, use_dask: bool = True, memory_limit: Optional[float] = None):
        self.use_ray = use_ray
        self.use_dask = use_dask
        self.memory_limit = memory_limit
        self.scaler = RobustScaler()
        self.models = {}
        self.best_params = {}
        
        if use_ray and not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load data with memory management."""
        try:
            if self.memory_limit is not None:
                available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
                if available_memory < self.memory_limit:
                    logging.warning(f"Available memory ({available_memory:.2f}GB) below limit ({self.memory_limit}GB)")
                    gc.collect()
            
            if self.use_dask:
                ddf = dd.read_csv(filepath)
                with ProgressBar():
                    return ddf.compute()
            return pd.read_csv(filepath)
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise
    
    def preprocess_data(self, data: pd.DataFrame, future_period: int = 1) -> pd.DataFrame:
        """Enhanced data preprocessing with parallel processing."""
        assert isinstance(data, pd.DataFrame), "data must be a DataFrame"
        assert 'price' in data.columns and 'volume' in data.columns, "data must contain columns 'price' and 'volume'"
        
        if self.use_dask:
            data = dd.from_pandas(data, npartitions=cpu_count())
        
        # Basic features
        data['price_change'] = data['price'].pct_change()
        data['volume_change'] = data['volume'].pct_change()
        
        # Enhanced moving averages
        for window in [5, 10, 20, 50, 100, 200]:
            data[f'ma_{window}'] = data['price'].rolling(window=window).mean()
            data[f'ema_{window}'] = data['price'].ewm(span=window, adjust=False).mean()
        
        # Moving average crossovers
        for short, long in [(5, 20), (10, 50), (20, 200)]:
            data[f'ma_cross_{short}_{long}'] = data[f'ma_{short}'] - data[f'ma_{long}']
            data[f'ema_cross_{short}_{long}'] = data[f'ema_{short}'] - data[f'ema_{long}']
        
        # Volatility features
        for window in [10, 20, 50]:
            data[f'std_{window}'] = data['price'].rolling(window=window).std()
            data[f'atr_{window}'] = self._calculate_atr(data['price'].values, data['price'].values, data['price'].values, window)
        
        # Momentum features
        for window in [5, 10, 20, 50]:
            data[f'momentum_{window}'] = data['price'] - data['price'].shift(window)
            data[f'roc_{window}'] = data['price'].pct_change(window)
            data[f'williams_r_{window}'] = self._calculate_williams_r(data['price'].values, window)
        
        # Volume features
        for window in [10, 20, 50]:
            data[f'volume_ma_{window}'] = data['volume'].rolling(window=window).mean()
            data[f'volume_std_{window}'] = data['volume'].rolling(window=window).std()
            data[f'obv_{window}'] = self._calculate_obv(data['price'].values, data['volume'].values, window)
        
        # Advanced technical indicators
        data['rsi'] = self._calculate_rsi(data['price'].values)
        data['macd'], data['macd_signal'], data['macd_hist'] = self._calculate_macd(data['price'].values)
        data['bollinger_upper'], data['bollinger_lower'] = self._calculate_bollinger_bands(data['price'].values)
        data['stochastic_k'], data['stochastic_d'] = self._calculate_stochastic(data['price'].values, data['price'].values, data['price'].values)
        data['adx'] = self._calculate_adx(data['price'].values, data['price'].values, data['price'].values)
        data['cci'] = self._calculate_cci(data['price'].values, data['price'].values, data['price'].values)
        data['mfi'] = self._calculate_mfi(data['price'].values, data['price'].values, data['price'].values, data['volume'].values)
        
        # Pattern recognition
        data['engulfing'] = self._detect_engulfing(data['price'].values, data['price'].values)
        data['hammer'] = self._detect_hammer(data['price'].values, data['price'].values)
        data['doji'] = self._detect_doji(data['price'].values, data['price'].values)
        
        # Create labels
        data['label'] = self.create_labels(data['price'], future_period)
        
        # Drop missing values
        data = data.dropna()
        
        if self.use_dask:
            with ProgressBar():
                return data.compute()
        return data
    
    def create_labels(self, price_series: pd.Series, future_period: int = 1) -> pd.Series:
        """Enhanced label creation with trend strength consideration."""
        future_price = price_series.shift(-future_period)
        price_change = (future_price - price_series) / price_series
        
        # Create labels based on price change magnitude
        labels = pd.Series(0, index=price_series.index)
        labels[price_change > 0.01] = 1  # 1% increase
        labels[price_change < -0.01] = -1  # 1% decrease
        
        return labels
    
    def optimize_hyperparameters(self, data: pd.DataFrame, features: List[str]) -> Dict:
        """Optimize hyperparameters using Optuna."""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
            }
            
            model = GradientBoostingClassifier(**params, random_state=42)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            scores = []
            for train_idx, val_idx in cv.split(data[features], data['label']):
                X_train, X_val = data[features].iloc[train_idx], data[features].iloc[val_idx]
                y_train, y_val = data['label'].iloc[train_idx], data['label'].iloc[val_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                score = f1_score(y_val, y_pred, average='weighted')
                scores.append(score)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize', sampler=TPESampler())
        study.optimize(objective, n_trials=50)
        
        return study.best_params
    
    def train_ensemble(self, data: pd.DataFrame, features: List[str]) -> Dict:
        """Train an ensemble of models with optimized parameters."""
        X = data[features]
        y = data['label']
        
        # Split data with time series consideration
        tscv = TimeSeriesSplit(n_splits=5)
        train_idx, val_idx = list(tscv.split(X))[-1]
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train base models
        base_models = {
            'gb': GradientBoostingClassifier(**self.best_params, random_state=42),
            'rf': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
            'svm': SVC(kernel='rbf', probability=True, random_state=42)
        }
        
        for name, model in base_models.items():
            model.fit(X_train_scaled, y_train)
            self.models[name] = model
        
        # Create and train voting classifier
        voting_clf = VotingClassifier(
            estimators=[(name, model) for name, model in base_models.items()],
            voting='soft'
        )
        voting_clf.fit(X_train_scaled, y_train)
        self.models['ensemble'] = voting_clf
        
        # Evaluate models
        results = {}
        for name, model in self.models.items():
            y_pred = model.predict(X_val_scaled)
            results[name] = {
                'f1': f1_score(y_val, y_pred, average='weighted'),
                'precision': precision_score(y_val, y_pred, average='weighted'),
                'recall': recall_score(y_val, y_pred, average='weighted'),
                'roc_auc': roc_auc_score(y_val, model.predict_proba(X_val_scaled), multi_class='ovr')
            }
        
        return results
    
    def save_models(self, directory: str):
        """Save all trained models."""
        for name, model in self.models.items():
            filepath = f"{directory}/{name}_model.pkl"
            joblib.dump(model, filepath)
            logging.info(f"Saved {name} model to {filepath}")
    
    def load_models(self, directory: str):
        """Load saved models."""
        for name in ['gb', 'rf', 'svm', 'ensemble']:
            filepath = f"{directory}/{name}_model.pkl"
            try:
                self.models[name] = joblib.load(filepath)
                logging.info(f"Loaded {name} model from {filepath}")
            except FileNotFoundError:
                logging.warning(f"Model file not found: {filepath}")
    
    def predict(self, data: pd.DataFrame, features: List[str]) -> Dict:
        """Make predictions using all models."""
        X = data[features]
        X_scaled = self.scaler.transform(X)
        
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = {
                'class': model.predict(X_scaled),
                'probability': model.predict_proba(X_scaled)
            }
        
        return predictions

def main():
    """Example usage of the enhanced training system."""
    try:
        trainer = AdvancedModelTrainer(use_ray=True, use_dask=True, memory_limit=4.0)
        
        # Load and preprocess data with advanced features
        data = trainer.load_data('data/historical_data.csv')
        processed_data = trainer.preprocess_data(data)
        
        # Define comprehensive feature set
        features = [
            # Basic features
            'price_change', 'volume_change',
            # Moving averages
            'ma_10', 'ma_50', 'ma_200',
            'ma_cross_5_20', 'ma_cross_10_50', 'ma_cross_20_200',
            # Volatility
            'std_10', 'std_50', 'atr_ma', 'bollinger_width',
            # Momentum
            'momentum_5', 'momentum_10', 'momentum_20',
            'roc_ma', 'momentum_ma', 'williams_r',
            # Technical indicators
            'rsi', 'macd', 'bollinger_upper', 'bollinger_lower',
            'stochastic_k', 'stochastic_d', 'adx', 'cci', 'mfi',
            # Volume analysis
            'volume_ma_ratio', 'obv_ma',
            # Advanced indicators
            'adx_ma', 'cci_ma', 'mfi_ma',
            # Pattern analysis
            'engulfing_strength', 'hammer_strength', 'doji_strength',
            # Feature interactions
            'rsi_volume', 'macd_volume', 'adx_volatility'
        ]
        
        # Optimize hyperparameters with multiple strategies
        best_params = trainer.optimize_hyperparameters(processed_data, features)
        trainer.best_params = best_params
        
        # Train advanced ensemble
        results = trainer.train_ensemble(processed_data, features)
        
        # Save models
        trainer.save_models('models')
        
        # Access results for each model
        for model_name, metrics in results.items():
            print(f"\n{model_name} Performance:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise
    finally:
        if ray.is_initialized():
            ray.shutdown()

if __name__ == "__main__":
    main()
