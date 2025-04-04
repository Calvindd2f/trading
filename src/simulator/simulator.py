import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path
import ray
from ray import tune
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import gc
import psutil
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
import talib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
from functools import partial
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam

# Configure logging
logging.basicConfig(level=logging.INFO)

class TradeType(Enum):
    """Types of trades that can be executed."""
    LONG = "long"
    SHORT = "short"
    CLOSE = "close"

class TradingStrategy(Enum):
    """Available trading strategies."""
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    TREND_FOLLOWING = "trend_following"
    VOLATILITY = "volatility"
    MOMENTUM_BREAKOUT = "momentum_breakout"
    TREND_MEAN_REVERSION = "trend_mean_reversion"
    VOLATILITY_BREAKOUT = "volatility_breakout"
    COMPOSITE = "composite"
    MACHINE_LEARNING = "machine_learning"

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

@dataclass
class Trade:
    """Represents a single trade."""
    timestamp: datetime
    trade_type: TradeType
    price: float
    amount: float
    balance: float
    positions: float
    prediction: int
    confidence: float
    strategy: TradingStrategy
    market_regime: MarketRegime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    strategy_combination: Optional[List[TradingStrategy]] = None

class AdvancedTradeSimulator:
    """
    Enhanced trading simulator with sophisticated market regime detection
    and adaptive strategy selection.
    """
    
    def __init__(
        self,
        model_path: str,
        data_path: str,
        initial_balance: float = 10000.0,
        max_position_size: float = 0.1,
        risk_per_trade: float = 0.02,
        use_ray: bool = True,
        use_dask: bool = True,
        memory_limit: Optional[float] = None,
        max_workers: int = 4
    ):
        self.model = self._load_model(model_path)
        self.data = self._load_data(data_path)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.max_position_size = max_position_size
        self.risk_per_trade = risk_per_trade
        self.positions = 0.0
        self.trades: List[Trade] = []
        self.use_ray = use_ray
        self.use_dask = use_dask
        self.memory_limit = memory_limit
        self.max_workers = max_workers
        
        # Initialize regime detection models
        self.regime_classifier = self._initialize_regime_classifier()
        self.regime_lstm = self._initialize_regime_lstm()
        
        # Strategy weights based on market regime
        self.strategy_weights = self._initialize_strategy_weights()
        
        # Strategy combinations
        self.strategy_combinations = self._initialize_strategy_combinations()
        
        if use_ray and not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
    
    def _initialize_regime_classifier(self) -> RandomForestClassifier:
        """Initialize the regime classification model."""
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    
    def _initialize_regime_lstm(self) -> Sequential:
        """Initialize the LSTM model for regime prediction."""
        model = Sequential([
            LSTM(64, input_shape=(10, 7), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(len(MarketRegime), activation='softmax')
        ])
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def _initialize_strategy_weights(self) -> Dict[MarketRegime, Dict[TradingStrategy, float]]:
        """Initialize strategy weights based on market regime."""
        return {
            MarketRegime.TRENDING_UP: {
                TradingStrategy.MOMENTUM: 0.3,
                TradingStrategy.TREND_FOLLOWING: 0.3,
                TradingStrategy.BREAKOUT: 0.2,
                TradingStrategy.MOMENTUM_BREAKOUT: 0.2
            },
            MarketRegime.TRENDING_DOWN: {
                TradingStrategy.MOMENTUM: 0.3,
                TradingStrategy.TREND_FOLLOWING: 0.3,
                TradingStrategy.BREAKOUT: 0.2,
                TradingStrategy.MOMENTUM_BREAKOUT: 0.2
            },
            MarketRegime.RANGING: {
                TradingStrategy.MEAN_REVERSION: 0.4,
                TradingStrategy.TREND_MEAN_REVERSION: 0.3,
                TradingStrategy.VOLATILITY: 0.3
            },
            MarketRegime.HIGH_VOLATILITY: {
                TradingStrategy.VOLATILITY: 0.4,
                TradingStrategy.VOLATILITY_BREAKOUT: 0.3,
                TradingStrategy.BREAKOUT: 0.3
            },
            MarketRegime.LOW_VOLATILITY: {
                TradingStrategy.MEAN_REVERSION: 0.4,
                TradingStrategy.TREND_MEAN_REVERSION: 0.3,
                TradingStrategy.MOMENTUM: 0.3
            },
            MarketRegime.MOMENTUM: {
                TradingStrategy.MOMENTUM: 0.5,
                TradingStrategy.MOMENTUM_BREAKOUT: 0.3,
                TradingStrategy.TREND_FOLLOWING: 0.2
            },
            MarketRegime.MEAN_REVERSION: {
                TradingStrategy.MEAN_REVERSION: 0.5,
                TradingStrategy.TREND_MEAN_REVERSION: 0.3,
                TradingStrategy.VOLATILITY: 0.2
            },
            MarketRegime.BREAKOUT: {
                TradingStrategy.BREAKOUT: 0.4,
                TradingStrategy.VOLATILITY_BREAKOUT: 0.3,
                TradingStrategy.MOMENTUM_BREAKOUT: 0.3
            },
            MarketRegime.COMPLEX: {
                TradingStrategy.COMPOSITE: 0.4,
                TradingStrategy.MACHINE_LEARNING: 0.3,
                TradingStrategy.TREND_FOLLOWING: 0.3
            }
        }
    
    def _initialize_strategy_combinations(self) -> Dict[TradingStrategy, List[TradingStrategy]]:
        """Initialize strategy combinations."""
        return {
            TradingStrategy.MOMENTUM_BREAKOUT: [
                TradingStrategy.MOMENTUM,
                TradingStrategy.BREAKOUT
            ],
            TradingStrategy.TREND_MEAN_REVERSION: [
                TradingStrategy.TREND_FOLLOWING,
                TradingStrategy.MEAN_REVERSION
            ],
            TradingStrategy.VOLATILITY_BREAKOUT: [
                TradingStrategy.VOLATILITY,
                TradingStrategy.BREAKOUT
            ],
            TradingStrategy.COMPOSITE: [
                TradingStrategy.MOMENTUM,
                TradingStrategy.TREND_FOLLOWING,
                TradingStrategy.MEAN_REVERSION
            ]
        }
    
    def calculate_regime_features(self, data: pd.DataFrame) -> RegimeFeatures:
        """Calculate comprehensive regime features."""
        # Trend features
        sma_20 = data['price'].rolling(window=20).mean()
        sma_50 = data['price'].rolling(window=50).mean()
        trend_strength = abs(sma_20 - sma_50) / sma_50
        
        # Volatility features
        volatility = data['price'].pct_change().rolling(window=20).std()
        avg_volatility = volatility.mean()
        
        # Momentum features
        rsi = talib.RSI(data['price'].values)
        macd, macd_signal, _ = talib.MACD(data['price'].values)
        momentum = (rsi - 50) / 50  # Normalized momentum
        
        # Mean reversion features
        bollinger_upper, bollinger_middle, bollinger_lower = talib.BBANDS(data['price'].values)
        zscore = (data['price'] - bollinger_middle) / (bollinger_upper - bollinger_lower)
        mean_reversion = -zscore  # Negative for mean reversion
        
        # Breakout features
        atr = talib.ATR(data['high'].values, data['low'].values, data['price'].values)
        breakout = (data['price'] - bollinger_upper) / atr
        
        # Volume features
        volume_ma = data['volume'].rolling(window=20).mean()
        volume = data['volume'] / volume_ma
        
        # Market correlation features
        market_correlation = data['price'].pct_change().rolling(window=20).corr(
            data['price'].pct_change().shift(1)
        )
        
        return RegimeFeatures(
            trend_strength=trend_strength.iloc[-1],
            volatility=volatility.iloc[-1] / avg_volatility,
            momentum=momentum[-1],
            mean_reversion=mean_reversion.iloc[-1],
            breakout=breakout.iloc[-1],
            volume=volume.iloc[-1],
            market_correlation=market_correlation.iloc[-1],
            regime_confidence=0.0  # Will be updated by classifier
        )
    
    def detect_market_regime(self, data: pd.DataFrame) -> Tuple[MarketRegime, float]:
        """Detect market regime using multiple methods."""
        # Calculate regime features
        features = self.calculate_regime_features(data)
        
        # Prepare features for classification
        feature_array = np.array([
            features.trend_strength,
            features.volatility,
            features.momentum,
            features.mean_reversion,
            features.breakout,
            features.volume,
            features.market_correlation
        ]).reshape(1, -1)
        
        # Scale features
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(feature_array)
        
        # Get regime prediction and confidence
        regime_probs = self.regime_classifier.predict_proba(scaled_features)[0]
        regime_idx = np.argmax(regime_probs)
        confidence = regime_probs[regime_idx]
        
        # Get regime from index
        regime = list(MarketRegime)[regime_idx]
        
        # Update regime confidence
        features.regime_confidence = confidence
        
        return regime, confidence
    
    def select_strategies(
        self,
        market_regime: MarketRegime,
        confidence: float,
        recent_performance: Dict[TradingStrategy, float]
    ) -> List[Tuple[TradingStrategy, float]]:
        """Select strategies based on market regime and performance."""
        # Get base strategies for the regime
        base_strategies = list(self.strategy_weights[market_regime].keys())
        
        # Add composite strategies based on regime
        if market_regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
            base_strategies.append(TradingStrategy.MOMENTUM_BREAKOUT)
        elif market_regime == MarketRegime.RANGING:
            base_strategies.append(TradingStrategy.TREND_MEAN_REVERSION)
        elif market_regime == MarketRegime.HIGH_VOLATILITY:
            base_strategies.append(TradingStrategy.VOLATILITY_BREAKOUT)
        elif market_regime == MarketRegime.COMPLEX:
            base_strategies.append(TradingStrategy.COMPOSITE)
            base_strategies.append(TradingStrategy.MACHINE_LEARNING)
        
        # Calculate strategy weights
        strategy_weights = []
        for strategy in base_strategies:
            # Base weight from regime
            base_weight = self.strategy_weights[market_regime][strategy]
            
            # Adjust weight based on confidence
            adjusted_weight = base_weight * confidence
            
            # Further adjust based on recent performance
            if strategy in recent_performance:
                performance_factor = 1 + recent_performance[strategy]
                adjusted_weight *= performance_factor
            
            strategy_weights.append((strategy, adjusted_weight))
        
        # Normalize weights
        total_weight = sum(weight for _, weight in strategy_weights)
        normalized_weights = [
            (strategy, weight / total_weight)
            for strategy, weight in strategy_weights
        ]
        
        return normalized_weights
    
    def calculate_recent_performance(
        self,
        trades: List[Trade],
        lookback_period: int = 20
    ) -> Dict[TradingStrategy, float]:
        """Calculate recent performance for each strategy."""
        if not trades:
            return {}
        
        # Get recent trades
        recent_trades = trades[-lookback_period:]
        
        # Calculate strategy performance
        strategy_performance = {}
        for strategy in TradingStrategy:
            strategy_trades = [t for t in recent_trades if t.strategy == strategy]
            if strategy_trades:
                pnl = sum(t.amount * t.price * (1 if t.trade_type == TradeType.CLOSE else -1)
                         for t in strategy_trades)
                strategy_performance[strategy] = pnl / len(strategy_trades)
            else:
                strategy_performance[strategy] = 0.0
        
        return strategy_performance
    
    async def execute_trade_async(
        self,
        prediction: int,
        confidence: float,
        row: pd.Series,
        volatility: float,
        strategy: TradingStrategy,
        market_regime: MarketRegime,
        strategy_weight: float
    ) -> None:
        """Execute a trade asynchronously with risk management."""
        price = row['price']
        position_size = self.calculate_position_size(price, volatility, strategy) * strategy_weight
        
        if prediction == 1 and self.balance >= position_size * price:  # Buy signal
            stop_loss, take_profit, trailing_stop = self.set_stop_loss_take_profit(
                price, TradeType.LONG, volatility, strategy
            )
            self.positions += position_size
            self.balance -= position_size * price
            
            # Determine strategy combination
            strategy_combination = None
            if strategy in self.strategy_combinations:
                strategy_combination = self.strategy_combinations[strategy]
            
            self.trades.append(Trade(
                timestamp=row['time'],
                trade_type=TradeType.LONG,
                price=price,
                amount=position_size,
                balance=self.balance,
                positions=self.positions,
                prediction=prediction,
                confidence=confidence,
                strategy=strategy,
                market_regime=market_regime,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trailing_stop=trailing_stop,
                risk_reward_ratio=abs(take_profit - price) / abs(price - stop_loss),
                strategy_combination=strategy_combination
            ))
            logging.info(
                f"Long: {position_size:.4f} at {price:.2f}, "
                f"SL: {stop_loss:.2f}, TP: {take_profit:.2f}, "
                f"Strategy: {strategy.value}, Regime: {market_regime.value}, "
                f"Weight: {strategy_weight:.2f}"
            )
        
        elif prediction == 0 and self.positions > 0:  # Sell signal
            self.balance += self.positions * price
            self.trades.append(Trade(
                timestamp=row['time'],
                trade_type=TradeType.CLOSE,
                price=price,
                amount=self.positions,
                balance=self.balance,
                positions=0.0,
                prediction=prediction,
                confidence=confidence,
                strategy=strategy,
                market_regime=market_regime
            ))
            logging.info(f"Sell: {self.positions:.4f} at {price:.2f}")
            self.positions = 0.0
    
    def simulate_trades(self) -> None:
        """Simulate trading with sophisticated strategy combinations."""
        logging.info("Starting advanced trade simulation...")
        
        # Calculate technical indicators
        self.data = self.calculate_technical_indicators(self.data)
        
        # Create event loop for async execution
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            for index, row in self.data.iterrows():
                # Detect market regime and confidence
                market_regime, confidence = self.detect_market_regime(self.data.iloc[:index+1])
                
                # Calculate recent performance
                recent_performance = self.calculate_recent_performance(self.trades)
                
                # Select strategies with weights
                strategies = self.select_strategies(market_regime, confidence, recent_performance)
                
                # Get model prediction and confidence
                features = row.drop(['time', 'price', 'volume', 'label'])
                prediction = self.model.predict([features])[0]
                model_confidence = np.max(self.model.predict_proba([features])[0])
                
                # Execute trades for each strategy
                for strategy, weight in strategies:
                    loop.run_until_complete(self.execute_trade_async(
                        prediction=prediction,
                        confidence=model_confidence,
                        row=row,
                        volatility=self.data['price'].pct_change().rolling(window=20).std().iloc[index],
                        strategy=strategy,
                        market_regime=market_regime,
                        strategy_weight=weight
                    ))
        finally:
            loop.close()
        
        self.save_results()
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not self.trades:
            return {}
        
        trades_df = pd.DataFrame([vars(trade) for trade in self.trades])
        trades_df['pnl'] = trades_df['amount'] * trades_df['price'] * (
            trades_df['trade_type'].apply(lambda x: 1 if x == TradeType.CLOSE else -1)
        )
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Advanced metrics
        returns = trades_df['pnl'].cumsum()
        max_drawdown = (returns - returns.cummax()).min()
        sharpe_ratio = returns.mean() / returns.std() if len(returns) > 1 else 0
        sortino_ratio = returns.mean() / returns[returns < 0].std() if len(returns[returns < 0]) > 0 else 0
        
        # Risk metrics
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if total_trades - winning_trades > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Strategy-specific metrics
        strategy_metrics = {}
        for strategy in TradingStrategy:
            strategy_trades = trades_df[trades_df['strategy'] == strategy]
            if len(strategy_trades) > 0:
                strategy_returns = strategy_trades['pnl'].cumsum()
                strategy_metrics[strategy.value] = {
                    'total_trades': len(strategy_trades),
                    'win_rate': len(strategy_trades[strategy_trades['pnl'] > 0]) / len(strategy_trades),
                    'total_return': strategy_returns.iloc[-1] if len(strategy_returns) > 0 else 0,
                    'sharpe_ratio': strategy_returns.mean() / strategy_returns.std() if len(strategy_returns) > 1 else 0
                }
        
        # Market regime metrics
        regime_metrics = {}
        for regime in MarketRegime:
            regime_trades = trades_df[trades_df['market_regime'] == regime]
            if len(regime_trades) > 0:
                regime_returns = regime_trades['pnl'].cumsum()
                regime_metrics[regime.value] = {
                    'total_trades': len(regime_trades),
                    'win_rate': len(regime_trades[regime_trades['pnl'] > 0]) / len(regime_trades),
                    'total_return': regime_returns.iloc[-1] if len(regime_returns) > 0 else 0,
                    'sharpe_ratio': regime_returns.mean() / regime_returns.std() if len(regime_returns) > 1 else 0
                }
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return': returns.iloc[-1],
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'final_balance': self.balance,
            'strategy_metrics': strategy_metrics,
            'regime_metrics': regime_metrics
        }
    
    def save_results(self) -> None:
        """Save simulation results and trade history."""
        # Save trade history
        trades_df = pd.DataFrame([vars(trade) for trade in self.trades])
        trades_df.to_csv('simulator_trades.csv', index=False)
        
        # Save performance metrics
        metrics = self.calculate_performance_metrics()
        pd.DataFrame([metrics]).to_csv('simulator_performance.csv', index=False)
        
        # Save detailed configuration
        config = {
            'initial_balance': self.initial_balance,
            'max_position_size': self.max_position_size,
            'risk_per_trade': self.risk_per_trade,
            'strategy_weights': {k.value: v for k, v in self.strategy_weights.items()},
            'final_balance': self.balance,
            'total_trades': len(self.trades),
            'performance_metrics': metrics
        }
        with open('simulator_config.json', 'w') as f:
            json.dump(config, f, indent=4)
        
        logging.info("Simulation results saved successfully")

def main():
    """Example usage of the enhanced simulator."""
    try:
        simulator = AdvancedTradeSimulator(
            model_path='models/ensemble_model.pkl',
            data_path='data/historical_data.csv',
            initial_balance=10000.0,
            max_position_size=0.1,
            risk_per_trade=0.02,
            use_ray=True,
            use_dask=True,
            memory_limit=4.0,
            max_workers=4
        )
        
        simulator.simulate_trades()
        metrics = simulator.calculate_performance_metrics()
        
        logging.info("\nOverall Performance Metrics:")
        for metric, value in metrics.items():
            if metric not in ['strategy_metrics', 'regime_metrics']:
                logging.info(f"{metric}: {value:.4f}")
        
        logging.info("\nStrategy-Specific Metrics:")
        for strategy, strategy_metrics in metrics['strategy_metrics'].items():
            logging.info(f"\n{strategy} Strategy:")
            for metric, value in strategy_metrics.items():
                logging.info(f"{metric}: {value:.4f}")
        
        logging.info("\nMarket Regime Metrics:")
        for regime, regime_metrics in metrics['regime_metrics'].items():
            logging.info(f"\n{regime} Regime:")
            for metric, value in regime_metrics.items():
                logging.info(f"{metric}: {value:.4f}")
        
    except Exception as e:
        logging.error(f"Error in simulation: {e}")
        raise
    finally:
        if ray.is_initialized():
            ray.shutdown()

if __name__ == "__main__":
    main()
