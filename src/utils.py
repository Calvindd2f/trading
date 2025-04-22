from builtins import isinstance, staticmethod
from copy import deepcopy
from locale import str
import logging
import sqlite3
from datetime import datetime
from functools import lru_cache
from typing import Optional, Tuple, List, Dict, Union, Any
import numpy as np
from numba import boolean, float64, int32, jit
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import talib
from scipy import stats, signal
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, FastICA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    Dropout,
    Input,
    Bidirectional,
    Attention,
    LayerNormalization,
)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import ray
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import gc
import psutil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
from functools import partial
import joblib
from pathlib import Path
import json
import optuna
from optuna.integration import TFKerasPruningCallback
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


class MarketRegime(Enum):
    """Market regime classifications."""

    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    COMPLEX = "complex"


@dataclass
class RegimeFeatures:
    """Features used for regime detection."""

    trend_strength: float
    volatility: float
    momentum: float
    mean_reversion: float
    breakout: float
    volume: float
    market_correlation: float
    regime_confidence: float
    fractal_dimension: float
    hurst_exponent: float
    entropy: float
    market_efficiency: float


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics."""

    total_trades: int
    win_rate: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    final_balance: float
    strategy_metrics: Dict[str, Dict[str, float]]
    regime_metrics: Dict[str, Dict[str, float]]
    risk_metrics: Dict[str, float]
    market_metrics: Dict[str, float]
    trade_quality_metrics: Dict[str, float]


class DataProcessor:
    """Enhanced data processing utilities."""

    @staticmethod
    def load_data(
        data_path: str,
        use_dask: bool = True,
        memory_limit: Optional[float] = None,
        preprocess: bool = True,
    ) -> pd.DataFrame:
        """Load data with memory management and preprocessing."""
        try:
            if memory_limit is not None:
                available_memory = psutil.virtual_memory().available / (
                    1024 * 1024 * 1024
                )
                if available_memory < memory_limit:
                    logging.warning(
                        f"Available memory ({available_memory:.2f}GB) below limit ({memory_limit}GB)"
                    )
                    gc.collect()

            if use_dask:
                ddf = dd.read_csv(data_path)
                with ProgressBar():
                    data = ddf.compute()
            else:
                data = pd.read_csv(data_path)

            if preprocess:
                data = DataProcessor.preprocess_data(data)

            return data
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    @staticmethod
    def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data with advanced techniques."""
        # Handle missing values
        data = data.interpolate(method="time")
        data = data.fillna(method="ffill").fillna(method="bfill")

        # Remove outliers using IQR
        for col in data.select_dtypes(include=[np.number]).columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            data[col] = data[col].clip(lower=Q1 - 1.5 * IQR, upper=Q3 + 1.5 * IQR)

        # Scale features
        scaler = RobustScaler()
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

        return data

    @staticmethod
    def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators."""
        # Momentum indicators
        data["rsi"] = talib.RSI(data["price"].values)
        data["macd"], data["macd_signal"], data["macd_hist"] = talib.MACD(
            data["price"].values
        )
        data["stochastic_k"], data["stochastic_d"] = talib.STOCH(
            data["high"].values, data["low"].values, data["price"].values
        )
        data["mfi"] = talib.MFI(
            data["high"].values,
            data["low"].values,
            data["price"].values,
            data["volume"].values,
        )
        data["cci"] = talib.CCI(
            data["high"].values, data["low"].values, data["price"].values
        )

        # Mean reversion indicators
        data["bollinger_upper"], data["bollinger_middle"], data["bollinger_lower"] = (
            talib.BBANDS(data["price"].values)
        )
        data["bollinger_width"] = (
            data["bollinger_upper"] - data["bollinger_lower"]
        ) / data["bollinger_middle"]
        data["zscore"] = stats.zscore(data["price"].values)
        data["williams_r"] = talib.WILLR(
            data["high"].values, data["low"].values, data["price"].values
        )

        # Trend indicators
        data["adx"] = talib.ADX(
            data["high"].values, data["low"].values, data["price"].values
        )
        data["aroon_up"], data["aroon_down"] = talib.AROON(
            data["high"].values, data["low"].values
        )
        data["tema"] = talib.TEMA(data["price"].values)
        data["kama"] = talib.KAMA(data["price"].values)

        # Volatility indicators
        data["atr"] = talib.ATR(
            data["high"].values, data["low"].values, data["price"].values
        )
        data["natr"] = talib.NATR(
            data["high"].values, data["low"].values, data["price"].values
        )
        data["trange"] = talib.TRANGE(
            data["high"].values, data["low"].values, data["price"].values
        )

        # Volume indicators
        data["obv"] = talib.OBV(data["price"].values, data["volume"].values)
        data["ad"] = talib.AD(
            data["high"].values,
            data["low"].values,
            data["price"].values,
            data["volume"].values,
        )
        data["adxr"] = talib.ADXR(
            data["high"].values, data["low"].values, data["price"].values
        )

        # Advanced indicators
        data["fractal_dimension"] = DataProcessor.calculate_fractal_dimension(
            data["price"].values
        )
        data["hurst_exponent"] = DataProcessor.calculate_hurst_exponent(
            data["price"].values
        )
        data["entropy"] = DataProcessor.calculate_entropy(data["price"].values)
        data["market_efficiency"] = DataProcessor.calculate_market_efficiency(
            data["price"].values
        )

        return data

    @staticmethod
    def calculate_fractal_dimension(prices: np.ndarray, window: int = 20) -> np.ndarray:
        """Calculate fractal dimension using Higuchi method."""
        n = len(prices)
        k_values = np.arange(1, window + 1)
        L = np.zeros_like(k_values, dtype=float)

        for i, k in enumerate(k_values):
            L[i] = np.sum(np.abs(np.diff(prices[::k]))) * (n - 1) / (k * (n // k))

        return -np.polyfit(np.log(k_values), np.log(L), 1)[0]

    @staticmethod
    def calculate_hurst_exponent(prices: np.ndarray, lags: List[int] = None) -> float:
        """Calculate Hurst exponent using rescaled range analysis."""
        if lags is None:
            lags = [2, 3, 4, 5, 6, 7, 8, 9, 10]

        tau = []
        laggedvar = []

        for lag in lags:
            # Calculate variance of lagged differences
            tau.append(lag)
            laggedvar.append(np.var(prices[lag:] - prices[:-lag]))

        # Calculate Hurst exponent
        m = np.polyfit(np.log(tau), np.log(laggedvar), 1)
        hurst = m[0] / 2.0

        return hurst

    @staticmethod
    def calculate_entropy(prices: np.ndarray, window: int = 20) -> np.ndarray:
        """Calculate sample entropy."""
        n = len(prices)
        entropy = np.zeros(n)

        for i in range(window, n):
            window_data = prices[i - window : i]
            hist, _ = np.histogram(window_data, bins="auto", density=True)
            entropy[i] = -np.sum(hist * np.log2(hist + 1e-10))

        return entropy

    @staticmethod
    def calculate_market_efficiency(prices: np.ndarray, window: int = 20) -> np.ndarray:
        """Calculate market efficiency ratio."""
        n = len(prices)
        efficiency = np.zeros(n)

        for i in range(window, n):
            window_data = prices[i - window : i]
            returns = np.diff(window_data)
            efficiency[i] = np.abs(np.sum(returns)) / np.sum(np.abs(returns))

        return efficiency

    @staticmethod
    def calculate_regime_features(data: pd.DataFrame) -> RegimeFeatures:
        """Calculate comprehensive regime features."""
        # Trend features
        sma_20 = data["price"].rolling(window=20).mean()
        sma_50 = data["price"].rolling(window=50).mean()
        trend_strength = abs(sma_20 - sma_50) / sma_50

        # Volatility features
        volatility = data["price"].pct_change().rolling(window=20).std()
        avg_volatility = volatility.mean()

        # Momentum features
        rsi = talib.RSI(data["price"].values)
        macd, macd_signal, _ = talib.MACD(data["price"].values)
        momentum = (rsi - 50) / 50  # Normalized momentum

        # Mean reversion features
        bollinger_upper, bollinger_middle, bollinger_lower = talib.BBANDS(
            data["price"].values
        )
        zscore = (data["price"] - bollinger_middle) / (
            bollinger_upper - bollinger_lower
        )
        mean_reversion = -zscore  # Negative for mean reversion

        # Breakout features
        atr = talib.ATR(data["high"].values, data["low"].values, data["price"].values)
        breakout = (data["price"] - bollinger_upper) / atr

        # Volume features
        volume_ma = data["volume"].rolling(window=20).mean()
        volume = data["volume"] / volume_ma

        # Market correlation features
        market_correlation = (
            data["price"]
            .pct_change()
            .rolling(window=20)
            .corr(data["price"].pct_change().shift(1))
        )

        # Advanced features
        fractal_dimension = DataProcessor.calculate_fractal_dimension(
            data["price"].values
        )
        hurst_exponent = DataProcessor.calculate_hurst_exponent(data["price"].values)
        entropy = DataProcessor.calculate_entropy(data["price"].values)
        market_efficiency = DataProcessor.calculate_market_efficiency(
            data["price"].values
        )

        return RegimeFeatures(
            trend_strength=trend_strength.iloc[-1],
            volatility=volatility.iloc[-1] / avg_volatility,
            momentum=momentum[-1],
            mean_reversion=mean_reversion.iloc[-1],
            breakout=breakout.iloc[-1],
            volume=volume.iloc[-1],
            market_correlation=market_correlation.iloc[-1],
            regime_confidence=0.0,
            fractal_dimension=fractal_dimension,
            hurst_exponent=hurst_exponent,
            entropy=entropy[-1],
            market_efficiency=market_efficiency[-1],
        )


class ModelUtils:
    """Enhanced model utilities."""

    def __init__(self):
        self.regime_classifier = ModelUtils.initialize_regime_classifier()
        self.ensemble_model = ModelUtils.create_ensemble_model()
        self.hyperparameters = ModelUtils.optimize_hyperparameters(
            self.regime_classifier,
            X=np.array(data[["feature1", "feature2", "feature3"]]),
            y=np.array(data["regime"]),
        )
        self.regime_classifier.set_params(**self.hyperparameters)
        self.ensemble_model.set_params(**self.hyperparameters)

    @staticmethod
    def initialize_regime_classifier() -> RandomForestClassifier:
        """Initialize the regime classification model."""
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )

    @staticmethod
    def initialize_regime_lstm() -> Sequential:
        """Initialize the LSTM model for regime prediction."""
        model = Sequential(
            [
                Bidirectional(LSTM(128, return_sequences=True), input_shape=(10, 12)),
                LayerNormalization(),
                Dropout(0.3),
                Bidirectional(LSTM(64, return_sequences=True)),
                LayerNormalization(),
                Dropout(0.3),
                Bidirectional(LSTM(32)),
                LayerNormalization(),
                Dropout(0.3),
                Dense(64, activation="relu"),
                Dense(32, activation="relu"),
                Dense(len(MarketRegime), activation="softmax"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    @staticmethod
    def create_ensemble_model() -> VotingClassifier:
        """Create an ensemble of models for regime prediction."""
        models = [
            ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
            ("gb", GradientBoostingClassifier(n_estimators=200, random_state=42)),
            ("lstm", ModelUtils.initialize_regime_lstm()),
        ]
        return VotingClassifier(estimators=models, voting="soft")

    @staticmethod
    def optimize_hyperparameters(
        model: Union[RandomForestClassifier, Sequential],
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 100,
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using Optuna."""

        def objective(trial):
            if isinstance(model, RandomForestClassifier):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                    "max_depth": trial.suggest_int("max_depth", 5, 20),
                    "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                }
                model_instance = clone(model)
                model_instance.set_params(**params)
            else:
                params = {
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 1e-5, 1e-2, log=True
                    ),
                    "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
                    "lstm_units": trial.suggest_int("lstm_units", 32, 256),
                }
                model_instance = ModelUtils.create_lstm_model(**params)

            scores = cross_val_score(model_instance, X, y, cv=5, scoring="accuracy")
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        return study.best_params

    @staticmethod
    def create_lstm_model(
        learning_rate: float = 0.001, dropout_rate: float = 0.3, lstm_units: int = 128
    ) -> Sequential:
        """Create an LSTM model with specified parameters."""
        model = Sequential(
            [
                Bidirectional(
                    LSTM(lstm_units, return_sequences=True), input_shape=(10, 12)
                ),
                LayerNormalization(),
                Dropout(dropout_rate),
                Bidirectional(LSTM(lstm_units // 2, return_sequences=True)),
                LayerNormalization(),
                Dropout(dropout_rate),
                Bidirectional(LSTM(lstm_units // 4)),
                LayerNormalization(),
                Dropout(dropout_rate),
                Dense(64, activation="relu"),
                Dense(32, activation="relu"),
                Dense(len(MarketRegime), activation="softmax"),
            ]
        )
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    @staticmethod
    def save_model(
        model: Union[RandomForestClassifier, Sequential, VotingClassifier], path: str
    ) -> None:
        """Save a model to disk."""
        try:
            if isinstance(model, Sequential):
                model.save(path)
            elif isinstance(model, VotingClassifier):
                joblib.dump(model, path)
            else:
                joblib.dump(model, path)
            logging.info(f"Model saved successfully to {path}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            raise

    @staticmethod
    def load_model(
        path: str,
    ) -> Union[RandomForestClassifier, Sequential, VotingClassifier]:
        """Load a model from disk."""
        try:
            if path.endswith(".h5"):
                return tf.keras.models.load_model(path)
            return joblib.load(path)
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise


class PerformanceAnalyzer:
    """Enhanced performance analysis utilities."""

    @staticmethod
    def calculate_performance_metrics(
        trades: List[Dict], initial_balance: float, final_balance: float
    ) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics."""
        if not trades:
            return PerformanceMetrics(
                total_trades=0,
                win_rate=0.0,
                total_return=0.0,
                max_drawdown=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                profit_factor=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                final_balance=final_balance,
                strategy_metrics={},
                regime_metrics={},
                risk_metrics={},
                market_metrics={},
                trade_quality_metrics={},
            )

        trades_df = pd.DataFrame(trades)
        trades_df["pnl"] = (
            trades_df["amount"]
            * trades_df["price"]
            * (trades_df["trade_type"].apply(lambda x: 1 if x == "close" else -1))
        )

        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df["pnl"] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # Advanced metrics
        returns = trades_df["pnl"].cumsum()
        max_drawdown = (returns - returns.cummax()).min()
        sharpe_ratio = returns.mean() / returns.std() if len(returns) > 1 else 0
        sortino_ratio = (
            returns.mean() / returns[returns < 0].std()
            if len(returns[returns < 0]) > 0
            else 0
        )

        avg_win = (
            trades_df[trades_df["pnl"] > 0]["pnl"].mean() if winning_trades > 0 else 0
        )
        avg_loss = (
            trades_df[trades_df["pnl"] < 0]["pnl"].mean()
            if total_trades - winning_trades > 0
            else 0
        )
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

        strategy_metrics = {}
        for strategy in trades_df["strategy"].unique():
            strategy_trades = trades_df[trades_df["strategy"] == strategy]
            if len(strategy_trades) > 0:
                strategy_returns = strategy_trades["pnl"].cumsum()
                strategy_metrics[strategy] = {
                    "total_trades": len(strategy_trades),
                    "win_rate": len(strategy_trades[strategy_trades["pnl"] > 0])
                    / len(strategy_trades),
                    "total_return": (
                        strategy_returns.iloc[-1] if len(strategy_returns) > 0 else 0
                    ),
                    "sharpe_ratio": (
                        strategy_returns.mean() / strategy_returns.std()
                        if len(strategy_returns) > 1
                        else 0
                    ),
                    "max_drawdown": (
                        strategy_returns - strategy_returns.cummax()
                    ).min(),
                    "profit_factor": (
                        abs(
                            strategy_trades[strategy_trades["pnl"] > 0]["pnl"].mean()
                            / strategy_trades[strategy_trades["pnl"] < 0]["pnl"].mean()
                        )
                        if len(strategy_trades[strategy_trades["pnl"] < 0]) > 0
                        else float("inf")
                    ),
                }

        regime_metrics = {}
        for regime in trades_df["market_regime"].unique():
            regime_trades = trades_df[trades_df["market_regime"] == regime]
            if len(regime_trades) > 0:
                regime_returns = regime_trades["pnl"].cumsum()
                regime_metrics[regime] = {
                    "total_trades": len(regime_trades),
                    "win_rate": len(regime_trades[regime_trades["pnl"] > 0])
                    / len(regime_trades),
                    "total_return": (
                        regime_returns.iloc[-1] if len(regime_returns) > 0 else 0
                    ),
                    "sharpe_ratio": (
                        regime_returns.mean() / regime_returns.std()
                        if len(regime_returns) > 1
                        else 0
                    ),
                    "max_drawdown": (regime_returns - regime_returns.cummax()).min(),
                    "profit_factor": (
                        abs(
                            regime_trades[regime_trades["pnl"] > 0]["pnl"].mean()
                            / regime_trades[regime_trades["pnl"] < 0]["pnl"].mean()
                        )
                        if len(regime_trades[regime_trades["pnl"] < 0]) > 0
                        else float("inf")
                    ),
                }

        risk_metrics = {
            "value_at_risk_95": np.percentile(returns, 5),
            "expected_shortfall": returns[returns <= np.percentile(returns, 5)].mean(),
            "calmar_ratio": (
                returns.mean() / abs(max_drawdown) if max_drawdown != 0 else 0
            ),
            "omega_ratio": (
                len(returns[returns > 0]) / len(returns[returns < 0])
                if len(returns[returns < 0]) > 0
                else float("inf")
            ),
            "tail_ratio": (
                abs(np.percentile(returns, 95) / np.percentile(returns, 5))
                if np.percentile(returns, 5) != 0
                else float("inf")
            ),
        }

        market_metrics = {
            "market_correlation": trades_df["pnl"].corr(
                trades_df["price"].pct_change()
            ),
            "market_timing": (
                len(
                    trades_df[
                        (trades_df["pnl"] > 0) & (trades_df["price"].pct_change() > 0)
                    ]
                )
                / len(trades_df[trades_df["price"].pct_change() > 0])
                if len(trades_df[trades_df["price"].pct_change() > 0]) > 0
                else 0
            ),
            "market_efficiency": 1
            - (trades_df["pnl"].std() / trades_df["price"].pct_change().std()),
            "market_adaptability": len(trades_df[trades_df["pnl"] > 0])
            / len(trades_df),
        }

        trade_quality_metrics = {
            "avg_holding_period": (
                trades_df["timestamp"].diff().dt.total_seconds() / 3600
            ).mean(),
            "win_loss_ratio": (
                abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
            ),
            "profit_consistency": len(trades_df[trades_df["pnl"] > 0]) / total_trades,
            "risk_reward_ratio": (
                abs(trades_df["take_profit"].mean() / trades_df["stop_loss"].mean())
                if "take_profit" in trades_df.columns
                and "stop_loss" in trades_df.columns
                else 0
            ),
            "trade_efficiency": trades_df["pnl"].sum()
            / (trades_df["amount"] * trades_df["price"]).sum(),
        }

        return PerformanceMetrics(
            total_trades=total_trades,
            win_rate=win_rate,
            total_return=returns.iloc[-1],
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            final_balance=final_balance,
            strategy_metrics=strategy_metrics,
            regime_metrics=regime_metrics,
            risk_metrics=risk_metrics,
            market_metrics=market_metrics,
            trade_quality_metrics=trade_quality_metrics,
        )

    @staticmethod
    def plot_performance_metrics(
        metrics: PerformanceMetrics, output_dir: str = "results"
    ) -> None:
        """Plot comprehensive performance metrics."""
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            plt.figure(figsize=(12, 6))
            strategy_returns = {
                k: v["total_return"] for k, v in metrics.strategy_metrics.items()
            }
            plt.bar(strategy_returns.keys(), strategy_returns.values())
            plt.title("Strategy Returns")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/strategy_returns.png")
            plt.close()

            plt.figure(figsize=(12, 6))
            regime_returns = {
                k: v["total_return"] for k, v in metrics.regime_metrics.items()
            }
            plt.bar(regime_returns.keys(), regime_returns.values())
            plt.title("Regime Returns")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/regime_returns.png")
            plt.close()

            plt.figure(figsize=(12, 6))
            risk_metrics = metrics.risk_metrics
            plt.bar(risk_metrics.keys(), risk_metrics.values())
            plt.title("Risk Metrics")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/risk_metrics.png")
            plt.close()

            plt.figure(figsize=(12, 6))
            quality_metrics = metrics.trade_quality_metrics
            plt.bar(quality_metrics.keys(), quality_metrics.values())
            plt.title("Trade Quality Metrics")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/trade_quality_metrics.png")
            plt.close()

            logging.info(f"Performance plots saved successfully to {output_dir}")
        except Exception as e:
            logging.error(f"Error plotting performance metrics: {e}")
            raise

    @staticmethod
    def save_performance_metrics(
        metrics: PerformanceMetrics,
        trades: List[Dict],
        config: Dict,
        output_dir: str = "results",
    ) -> None:
        """Save performance metrics and trade history."""
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            trades_df = pd.DataFrame(trades)
            trades_df.to_csv(f"{output_dir}/trades.csv", index=False)

            metrics_dict = {
                "total_trades": metrics.total_trades,
                "win_rate": metrics.win_rate,
                "total_return": metrics.total_return,
                "max_drawdown": metrics.max_drawdown,
                "sharpe_ratio": metrics.sharpe_ratio,
                "sortino_ratio": metrics.sortino_ratio,
                "profit_factor": metrics.profit_factor,
                "avg_win": metrics.avg_win,
                "avg_loss": metrics.avg_loss,
                "final_balance": metrics.final_balance,
                "strategy_metrics": metrics.strategy_metrics,
                "regime_metrics": metrics.regime_metrics,
                "risk_metrics": metrics.risk_metrics,
                "market_metrics": metrics.market_metrics,
                "trade_quality_metrics": metrics.trade_quality_metrics,
            }

            with open(f"{output_dir}/metrics.json", "w") as f:
                json.dump(metrics_dict, f, indent=4)

            with open(f"{output_dir}/config.json", "w") as f:
                json.dump(config, f, indent=4)

            PerformanceAnalyzer.plot_performance_metrics(metrics, output_dir)

            logging.info(f"Results saved successfully to {output_dir}")
        except Exception as e:
            logging.error(f"Error saving results: {e}")
            raise


class ParallelProcessor:
    """Enhanced parallel processing utilities."""

    @staticmethod
    def initialize_ray() -> None:
        """Initialize Ray for distributed computing."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)

    @staticmethod
    def shutdown_ray() -> None:
        """Shutdown Ray."""
        if ray.is_initialized():
            ray.shutdown()

    @staticmethod
    async def process_chunk_async(
        chunk: pd.DataFrame, processor_func: callable, **kwargs
    ) -> pd.DataFrame:
        """Process a data chunk asynchronously."""
        try:
            return await asyncio.to_thread(processor_func, chunk, **kwargs)
        except Exception as e:
            logging.error(f"Error processing chunk: {e}")
            raise

    @staticmethod
    def process_parallel(
        data: pd.DataFrame,
        processor_func: callable,
        chunk_size: int = 1000,
        max_workers: int = 4,
        **kwargs,
    ) -> pd.DataFrame:
        """Process data in parallel using multiple workers."""
        try:
            chunks = [data[i : i + chunk_size] for i in range(0, len(data), chunk_size)]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(processor_func, chunk, **kwargs) for chunk in chunks
                ]

                results = []
                for future in futures:
                    results.append(future.result())

            return pd.concat(results, ignore_index=True)
        except Exception as e:
            logging.error(f"Error in parallel processing: {e}")
            raise


def main():
    """Example usage of the enhanced utilities."""
    try:
        data_processor = DataProcessor()

        data = data_processor.load_data(
            data_path="data/historical_data.csv",
            use_dask=True,
            memory_limit=4.0,
            preprocess=True,
        )
        data = data_processor.calculate_technical_indicators(data)

        model_utils = ModelUtils()
        regime_classifier = model_utils.initialize_regime_classifier()
        regime_lstm = model_utils.initialize_regime_lstm()
        ensemble_model = model_utils.create_ensemble_model()

        best_params = model_utils.optimize_hyperparameters(
            regime_classifier,
            data.drop(["time", "price", "volume", "label"], axis=1).values,
            data["label"].values,
        )

        model_utils.save_model(regime_classifier, "models/regime_classifier.pkl")
        model_utils.save_model(regime_lstm, "models/regime_lstm.h5")
        model_utils.save_model(ensemble_model, "models/ensemble_model.pkl")

        parallel_processor = ParallelProcessor()
        processed_data = parallel_processor.process_parallel(
            data=data,
            processor_func=data_processor.calculate_technical_indicators,
            chunk_size=1000,
            max_workers=4,
        )

        performance_analyzer = PerformanceAnalyzer()
        metrics = performance_analyzer.calculate_performance_metrics(
            trades=[],
            initial_balance=10000.0,
            final_balance=12000.0,
        )

        performance_analyzer.save_performance_metrics(
            metrics=metrics,
            trades=[],
            config={
                "initial_balance": 10000.0,
                "max_position_size": 0.1,
                "risk_per_trade": 0.02,
            },
            output_dir="results",
        )

    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise
    finally:
        ParallelProcessor.shutdown_ray()


if __name__ == "__main__":
    main()
