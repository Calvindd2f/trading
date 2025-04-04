import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_validate, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, PowerTransformer, QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, RFE, SelectKBest, mutual_info_classif, f_classif, chi2, VarianceThreshold
from sklearn.decomposition import PCA, FastICA, TruncatedSVD, KernelPCA
from sklearn.cluster import FeatureAgglomeration
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict, List, Tuple, Optional, Any, Union
import optuna
from optuna.integration import OptunaSearchCV
from optuna.pruners import MedianPruner, SuccessiveHalvingPruner, HyperbandPruner
from optuna.samplers import TPESampler, RandomSampler, CmaEsSampler, NSGAIISampler
from joblib import Parallel, delayed, cpu_count, Memory
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining, HyperBandScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.integration.keras import TuneReportCallback
from ray.tune.integration.xgboost import TuneReportCheckpointCallback
from ray.tune.integration.lightgbm import TuneReportCheckpointCallback as LGBMTuneReportCheckpointCallback
import dask
from dask.distributed import Client, LocalCluster
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM, GRU, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class AdvancedTradingFeatureSelector(BaseEstimator, TransformerMixin):
    """Enhanced feature selector with advanced trading-specific methods."""
    
    def __init__(self, method: str = 'importance', n_features: int = 10, 
                 scoring: str = 'f1', threshold: float = None,
                 time_window: int = 20):
        self.method = method
        self.n_features = n_features
        self.scoring = scoring
        self.threshold = threshold
        self.time_window = time_window
        self.selector = None
        self.selected_features = None
        self.memory = Memory(location='./cache', verbose=0)
    
    def _calculate_technical_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced technical indicators as features."""
        # Price-based features
        X['returns'] = X['close'].pct_change()
        X['log_returns'] = np.log1p(X['returns'])
        X['volatility'] = X['returns'].rolling(self.time_window).std()
        X['volume_ma'] = X['volume'].rolling(self.time_window).mean()
        X['price_ma'] = X['close'].rolling(self.time_window).mean()
        X['price_ema'] = X['close'].ewm(span=self.time_window).mean()
        
        # Momentum indicators
        X['rsi'] = self._calculate_rsi(X['close'])
        X['macd'] = self._calculate_macd(X['close'])
        X['stochastic'] = self._calculate_stochastic(X['high'], X['low'], X['close'])
        X['williams_r'] = self._calculate_williams_r(X['high'], X['low'], X['close'])
        X['roc'] = self._calculate_roc(X['close'])
        X['mfi'] = self._calculate_mfi(X['high'], X['low'], X['close'], X['volume'])
        
        # Trend indicators
        X['adx'] = self._calculate_adx(X['high'], X['low'], X['close'])
        X['atr'] = self._calculate_atr(X['high'], X['low'], X['close'])
        X['bollinger_upper'], X['bollinger_lower'] = self._calculate_bollinger_bands(X['close'])
        X['ichimoku_tenkan'], X['ichimoku_kijun'] = self._calculate_ichimoku(X['high'], X['low'])
        
        # Volume indicators
        X['obv'] = self._calculate_obv(X['close'], X['volume'])
        X['vwap'] = self._calculate_vwap(X['close'], X['volume'])
        X['volume_ratio'] = X['volume'] / X['volume_ma']
        
        # Volatility indicators
        X['keltner_upper'], X['keltner_lower'] = self._calculate_keltner_channels(X['high'], X['low'], X['close'])
        X['donchian_upper'], X['donchian_lower'] = self._calculate_donchian_channels(X['high'], X['low'])
        
        # Pattern recognition
        X['engulfing'] = self._detect_engulfing_pattern(X['open'], X['close'])
        X['hammer'] = self._detect_hammer_pattern(X['open'], X['high'], X['low'], X['close'])
        X['doji'] = self._detect_doji_pattern(X['open'], X['high'], X['low'], X['close'])
        
        return X
    
    @staticmethod
    def _calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Williams %R indicator."""
        highest_high = high.rolling(window=window).max()
        lowest_low = low.rolling(window=window).min()
        return -100 * (highest_high - close) / (highest_high - lowest_low)
    
    @staticmethod
    def _calculate_roc(close: pd.Series, window: int = 12) -> pd.Series:
        """Calculate Rate of Change indicator."""
        return (close - close.shift(window)) / close.shift(window) * 100
    
    @staticmethod
    def _calculate_mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Money Flow Index."""
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        positive_mf = positive_flow.rolling(window=window).sum()
        negative_mf = negative_flow.rolling(window=window).sum()
        
        mfi = 100 - (100 / (1 + positive_mf / negative_mf))
        return mfi
    
    @staticmethod
    def _calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Average Directional Index."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
        
        plus_di = 100 * plus_dm.rolling(window=window).mean() / atr
        minus_di = 100 * minus_dm.rolling(window=window).mean() / atr
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        return dx.rolling(window=window).mean()
    
    @staticmethod
    def _calculate_bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = close.rolling(window=window).mean()
        std = close.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, lower_band
    
    @staticmethod
    def _calculate_ichimoku(high: pd.Series, low: pd.Series, tenkan: int = 9, kijun: int = 26) -> Tuple[pd.Series, pd.Series]:
        """Calculate Ichimoku Cloud components."""
        tenkan_sen = (high.rolling(window=tenkan).max() + low.rolling(window=tenkan).min()) / 2
        kijun_sen = (high.rolling(window=kijun).max() + low.rolling(window=kijun).min()) / 2
        return tenkan_sen, kijun_sen
    
    @staticmethod
    def _calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate On-Balance Volume."""
        obv = volume.copy()
        obv[close < close.shift(1)] = -obv
        return obv.cumsum()
    
    @staticmethod
    def _calculate_vwap(close: pd.Series, volume: pd.Series) -> pd.Series:
        """Calculate Volume Weighted Average Price."""
        return (close * volume).cumsum() / volume.cumsum()
    
    @staticmethod
    def _calculate_keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20, multiplier: float = 2.0) -> Tuple[pd.Series, pd.Series]:
        """Calculate Keltner Channels."""
        typical_price = (high + low + close) / 3
        atr = abs(high - low).rolling(window=window).mean()
        upper_band = typical_price + (multiplier * atr)
        lower_band = typical_price - (multiplier * atr)
        return upper_band, lower_band
    
    @staticmethod
    def _calculate_donchian_channels(high: pd.Series, low: pd.Series, window: int = 20) -> Tuple[pd.Series, pd.Series]:
        """Calculate Donchian Channels."""
        upper_band = high.rolling(window=window).max()
        lower_band = low.rolling(window=window).min()
        return upper_band, lower_band
    
    @staticmethod
    def _detect_engulfing_pattern(open: pd.Series, close: pd.Series) -> pd.Series:
        """Detect Engulfing Pattern."""
        bullish = (close > open) & (close.shift(1) < open.shift(1)) & (close > open.shift(1)) & (open < close.shift(1))
        bearish = (close < open) & (close.shift(1) > open.shift(1)) & (close < open.shift(1)) & (open > close.shift(1))
        return pd.Series(np.where(bullish, 1, np.where(bearish, -1, 0)), index=open.index)
    
    @staticmethod
    def _detect_hammer_pattern(open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Detect Hammer Pattern."""
        body = abs(close - open)
        upper_shadow = high - np.maximum(open, close)
        lower_shadow = np.minimum(open, close) - low
        is_hammer = (lower_shadow > 2 * body) & (upper_shadow < body)
        return pd.Series(np.where(is_hammer, 1, 0), index=open.index)
    
    @staticmethod
    def _detect_doji_pattern(open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Detect Doji Pattern."""
        body = abs(close - open)
        total_range = high - low
        is_doji = (body <= 0.1 * total_range)
        return pd.Series(np.where(is_doji, 1, 0), index=open.index)

class AdvancedTradingModelTuner:
    """Enhanced model tuning system with advanced trading-specific optimizations."""
    
    def __init__(self, n_jobs: int = -1, cv: int = 5, use_ray: bool = False, 
                 use_dask: bool = False, time_series: bool = True):
        self.n_jobs = n_jobs if n_jobs != -1 else cpu_count()
        self.cv = TimeSeriesSplit(n_splits=cv) if time_series else StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        self.use_ray = use_ray
        self.use_dask = use_dask
        
        if use_ray and not ray.is_initialized():
            ray.init(num_cpus=self.n_jobs)
        
        if use_dask:
            self.cluster = LocalCluster(n_workers=self.n_jobs)
            self.client = Client(self.cluster)
        
        self.scorers = {
            'f1': make_scorer(f1_score),
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score),
            'roc_auc': make_scorer(roc_auc_score),
            'average_precision': make_scorer(average_precision_score),
            'balanced_accuracy': make_scorer(balanced_accuracy_score),
            'matthews_corrcoef': make_scorer(matthews_corrcoef),
            'cohen_kappa': make_scorer(cohen_kappa_score)
        }
        
        self.best_models = {}
        self.best_params = {}
        self.best_scores = {}
        self.feature_importance = {}
        self.model_metrics = {}
        self.trading_metrics = {}
        self.ensemble_models = {}
    
    def _create_advanced_ensemble(self, base_models: Dict[str, Any]) -> Dict[str, Any]:
        """Create advanced ensemble models."""
        return {
            'voting': VotingClassifier(
                estimators=[(name, model) for name, model in base_models.items()],
                voting='soft',
                n_jobs=self.n_jobs
            ),
            'stacking': StackingClassifier(
                estimators=[(name, model) for name, model in base_models.items()],
                final_estimator=LogisticRegression(),
                n_jobs=self.n_jobs
            ),
            'bagging': BaggingClassifier(
                base_estimator=RandomForestClassifier(),
                n_estimators=100,
                n_jobs=self.n_jobs
            )
        }
    
    def _create_advanced_neural_network(self) -> Sequential:
        """Create an advanced neural network model for trading."""
        model = Sequential([
            Conv1D(64, kernel_size=3, activation='relu', input_shape=(None, 1)),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),
            LSTM(64, return_sequences=True),
            BatchNormalization(),
            Dropout(0.2),
            LSTM(32),
            BatchNormalization(),
            Dropout(0.2),
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def _calculate_advanced_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  returns: np.ndarray) -> Dict[str, float]:
        """Calculate advanced trading performance metrics."""
        # Basic metrics
        cm = confusion_matrix(y_true, y_pred)
        win_rate = (cm[1, 1] + cm[0, 0]) / cm.sum()
        profit_factor = cm[1, 1] / (cm[0, 1] + 1e-6)
        
        # Risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        max_drawdown = self._calculate_max_drawdown(returns)
        calmar_ratio = self._calculate_calmar_ratio(returns)
        
        # Trade metrics
        avg_trade_return = np.mean(returns)
        std_trade_return = np.std(returns)
        win_loss_ratio = self._calculate_win_loss_ratio(returns)
        avg_win_size = self._calculate_avg_win_size(returns)
        avg_loss_size = self._calculate_avg_loss_size(returns)
        
        # Time-based metrics
        holding_period = self._calculate_avg_holding_period(y_pred)
        trade_frequency = self._calculate_trade_frequency(y_pred)
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'avg_trade_return': avg_trade_return,
            'std_trade_return': std_trade_return,
            'win_loss_ratio': win_loss_ratio,
            'avg_win_size': avg_win_size,
            'avg_loss_size': avg_loss_size,
            'holding_period': holding_period,
            'trade_frequency': trade_frequency
        }
    
    @staticmethod
    def _calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.01) -> float:
        """Calculate Sortino ratio."""
        excess_returns = returns - risk_free_rate / 252
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0:
            return 0
        downside_std = np.std(downside_returns)
        return np.mean(excess_returns) / downside_std if downside_std != 0 else 0
    
    @staticmethod
    def _calculate_calmar_ratio(returns: np.ndarray) -> float:
        """Calculate Calmar ratio."""
        max_drawdown = np.max(np.maximum.accumulate(returns) - returns)
        return np.mean(returns) / max_drawdown if max_drawdown != 0 else 0
    
    @staticmethod
    def _calculate_win_loss_ratio(returns: np.ndarray) -> float:
        """Calculate win/loss ratio."""
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]
        if len(losing_trades) == 0:
            return float('inf')
        return np.mean(winning_trades) / abs(np.mean(losing_trades))
    
    @staticmethod
    def _calculate_avg_win_size(returns: np.ndarray) -> float:
        """Calculate average winning trade size."""
        winning_trades = returns[returns > 0]
        return np.mean(winning_trades) if len(winning_trades) > 0 else 0
    
    @staticmethod
    def _calculate_avg_loss_size(returns: np.ndarray) -> float:
        """Calculate average losing trade size."""
        losing_trades = returns[returns < 0]
        return np.mean(losing_trades) if len(losing_trades) > 0 else 0
    
    @staticmethod
    def _calculate_avg_holding_period(predictions: np.ndarray) -> float:
        """Calculate average holding period."""
        changes = np.diff(predictions)
        holding_periods = np.where(changes != 0)[0]
        if len(holding_periods) < 2:
            return 0
        return np.mean(np.diff(holding_periods))
    
    @staticmethod
    def _calculate_trade_frequency(predictions: np.ndarray) -> float:
        """Calculate trade frequency."""
        changes = np.diff(predictions)
        return np.sum(changes != 0) / len(predictions)
    
    def tune_models(self, X: pd.DataFrame, y: pd.Series, 
                   method: str = 'ray', feature_selection: str = 'importance') -> Dict[str, Any]:
        """Tune models using specified optimization method."""
        results = {}
        
        # Feature selection
        if feature_selection:
            selector = AdvancedTradingFeatureSelector(
                method=feature_selection,
                n_features=min(20, X.shape[1]),
                time_window=20
            )
            X = selector.fit_transform(X, y)
            logging.info(f"Selected {X.shape[1]} features using {feature_selection}")
        
        # Tune base models
        base_models = {}
        for model_name, model in self._get_trading_models().items():
            logging.info(f"Tuning {model_name} model...")
            
            if self.use_ray:
                best_model, best_params, best_score = self._optimize_with_ray(model_name, X, y)
            elif self.use_dask:
                best_model, best_params, best_score = self._optimize_with_dask(model_name, X, y)
            else:
                param_grid = self._get_trading_param_grids()[model_name]
                pipeline = self._create_trading_pipeline(model)
                
                if method == 'grid':
                    search = GridSearchCV(
                        pipeline, param_grid, cv=self.cv, scoring='f1',
                        n_jobs=self.n_jobs, verbose=1
                    )
                else:  # random
                    search = RandomizedSearchCV(
                        pipeline, param_grid, cv=self.cv, scoring='f1',
                        n_jobs=self.n_jobs, n_iter=50, random_state=42, verbose=1
                    )
                
                search.fit(X, y)
                best_model = search.best_estimator_
                best_params = search.best_params_
                best_score = search.best_score_
            
            base_models[model_name] = best_model
            self.best_models[model_name] = best_model
            self.best_params[model_name] = best_params
            self.best_scores[model_name] = best_score
            
            # Calculate advanced metrics
            y_pred = best_model.predict(X)
            returns = np.where(y_pred == y, 1, -1)
            metrics = self._calculate_advanced_metrics(y, y_pred, returns)
            self.model_metrics[model_name] = metrics
            
            results[model_name] = {
                'model': best_model,
                'params': best_params,
                'score': best_score,
                'metrics': metrics
            }
            
            logging.info(f"Best {model_name} score: {best_score:.4f}")
            logging.info(f"Best {model_name} parameters: {best_params}")
            logging.info(f"Model metrics: {metrics}")
        
        # Tune ensemble models
        ensemble_models = self._create_advanced_ensemble(base_models)
        for ensemble_name, ensemble_model in ensemble_models.items():
            logging.info(f"Tuning {ensemble_name} ensemble...")
            
            if self.use_ray:
                best_ensemble, best_params, best_score = self._optimize_with_ray(ensemble_name, X, y)
            else:
                best_ensemble = ensemble_model
                best_ensemble.fit(X, y)
                best_params = {}
                best_score = best_ensemble.score(X, y)
            
            self.ensemble_models[ensemble_name] = best_ensemble
            self.best_models[ensemble_name] = best_ensemble
            self.best_params[ensemble_name] = best_params
            self.best_scores[ensemble_name] = best_score
            
            # Calculate advanced metrics for ensemble
            y_pred = best_ensemble.predict(X)
            returns = np.where(y_pred == y, 1, -1)
            metrics = self._calculate_advanced_metrics(y, y_pred, returns)
            self.model_metrics[ensemble_name] = metrics
            
            results[ensemble_name] = {
                'model': best_ensemble,
                'params': best_params,
                'score': best_score,
                'metrics': metrics
            }
            
            logging.info(f"Best {ensemble_name} score: {best_score:.4f}")
            logging.info(f"Best {ensemble_name} parameters: {best_params}")
            logging.info(f"Ensemble metrics: {metrics}")
        
        return results

def main():
    """Main function to demonstrate advanced model tuning."""
    try:
        # Load data
        data = pd.read_csv('data/historical_data.csv')
        X = data.drop('label', axis=1)
        y = data['label']
        
        # Initialize advanced tuner with Ray and Dask support
        tuner = AdvancedTradingModelTuner(
            n_jobs=-1, 
            cv=5, 
            use_ray=True,
            use_dask=True,
            time_series=True
        )
        
        # Tune models using Ray with advanced feature selection
        results = tuner.tune_models(
            X, y, 
            method='ray',
            feature_selection='importance'
        )
        
        # Get feature importance
        feature_importance = tuner.get_feature_importance('rf')
        if feature_importance is not None:
            logging.info("\nFeature Importance:")
            logging.info(feature_importance.head(10))
        
        # Get best model and metrics
        best_model, best_model_name = tuner.get_best_model()
        metrics = tuner.get_model_metrics()
        
        logging.info(f"\nBest model: {best_model_name}")
        logging.info(f"Best score: {tuner.best_scores[best_model_name]:.4f}")
        logging.info(f"Model metrics: {metrics[best_model_name]}")
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise
    finally:
        if ray.is_initialized():
            ray.shutdown()
        if hasattr(tuner, 'client'):
            tuner.client.close()
            tuner.cluster.close()

if __name__ == "__main__":
    main()
