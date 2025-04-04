import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator, ROCIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator, ADXIndicator, IchimokuIndicator
from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannels
from ta.volume import VolumeWeightedAveragePrice, OnBalanceVolume, ChaikinMoneyFlow
from ta.others import DailyReturnIndicator, CumulativeReturnIndicator
from typing import List, Dict, Tuple, Optional, Any
import numba
from functools import lru_cache
import warnings
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import threading
import queue
import time
from datetime import datetime
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@numba.jit(nopython=True)
def calculate_price_features(prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate price features using Numba for performance."""
    n = len(prices)
    momentum = np.zeros(n)
    volatility = np.zeros(n)
    returns = np.zeros(n)
    
    for i in range(1, n):
        returns[i] = (prices[i] - prices[i-1]) / prices[i-1]
        momentum[i] = prices[i] - prices[max(0, i-4)]
        volatility[i] = np.std(prices[max(0, i-10):i+1])
    
    return momentum, volatility, returns

class TradingSignal:
    """Class to handle trading signal analysis and generation."""
    
    def __init__(self):
        self.signals = {
            'trend': None,
            'momentum': None,
            'volatility': None,
            'volume': None,
            'strength': 0.0
        }
        self.confidence = 0.0
        self.timestamp = None
    
    def update(self, data: Dict[str, Any]) -> None:
        """Update signal with new data."""
        self.timestamp = datetime.now()
        self._analyze_trend(data)
        self._analyze_momentum(data)
        self._analyze_volatility(data)
        self._analyze_volume(data)
        self._calculate_strength()
    
    def _analyze_trend(self, data: Dict[str, Any]) -> None:
        """Analyze trend signals."""
        adx = data.get('adx', 0)
        ichimoku = data.get('ichimoku', {})
        
        if adx > 25:
            if ichimoku.get('tenkan_sen', 0) > ichimoku.get('kijun_sen', 0):
                self.signals['trend'] = 'bullish'
            else:
                self.signals['trend'] = 'bearish'
        else:
            self.signals['trend'] = 'neutral'
    
    def _analyze_momentum(self, data: Dict[str, Any]) -> None:
        """Analyze momentum signals."""
        rsi = data.get('rsi', 50)
        stoch = data.get('stoch', {})
        
        if rsi > 70:
            self.signals['momentum'] = 'overbought'
        elif rsi < 30:
            self.signals['momentum'] = 'oversold'
        else:
            self.signals['momentum'] = 'neutral'
    
    def _analyze_volatility(self, data: Dict[str, Any]) -> None:
        """Analyze volatility signals."""
        atr = data.get('atr', 0)
        bb = data.get('bollinger', {})
        
        if atr > bb.get('upper', 0):
            self.signals['volatility'] = 'high'
        elif atr < bb.get('lower', 0):
            self.signals['volatility'] = 'low'
        else:
            self.signals['volatility'] = 'normal'
    
    def _analyze_volume(self, data: Dict[str, Any]) -> None:
        """Analyze volume signals."""
        obv = data.get('obv', 0)
        cmf = data.get('cmf', 0)
        
        if obv > 0 and cmf > 0:
            self.signals['volume'] = 'bullish'
        elif obv < 0 and cmf < 0:
            self.signals['volume'] = 'bearish'
        else:
            self.signals['volume'] = 'neutral'
    
    def _calculate_strength(self) -> None:
        """Calculate overall signal strength."""
        strength = 0.0
        if self.signals['trend'] == 'bullish':
            strength += 0.3
        elif self.signals['trend'] == 'bearish':
            strength -= 0.3
        
        if self.signals['momentum'] == 'oversold':
            strength += 0.2
        elif self.signals['momentum'] == 'overbought':
            strength -= 0.2
        
        if self.signals['volume'] == 'bullish':
            strength += 0.2
        elif self.signals['volume'] == 'bearish':
            strength -= 0.2
        
        self.signals['strength'] = strength
        self.confidence = abs(strength)

class RealTimeVisualizer:
    """Class to handle real-time visualization updates."""
    
    def __init__(self):
        self.fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Price', 'Volume', 'Indicators', 'Signals'),
            row_heights=[0.4, 0.2, 0.2, 0.2]
        )
        self.data_queue = queue.Queue()
        self.update_thread = None
        self.running = False
    
    def start(self):
        """Start the visualization update thread."""
        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.start()
    
    def stop(self):
        """Stop the visualization update thread."""
        self.running = False
        if self.update_thread:
            self.update_thread.join()
    
    def _update_loop(self):
        """Main update loop for real-time visualization."""
        while self.running:
            try:
                data = self.data_queue.get(timeout=1)
                self._update_plot(data)
            except queue.Empty:
                continue
    
    def _update_plot(self, data: Dict[str, Any]):
        """Update the plot with new data."""
        # Update price chart
        self.fig.add_trace(
            go.Candlestick(
                x=[data['timestamp']],
                open=[data['open']],
                high=[data['high']],
                low=[data['low']],
                close=[data['close']]
            ),
            row=1, col=1
        )
        
        # Update volume chart
        self.fig.add_trace(
            go.Bar(
                x=[data['timestamp']],
                y=[data['volume']],
                name='Volume'
            ),
            row=2, col=1
        )
        
        # Update indicators
        self.fig.add_trace(
            go.Scatter(
                x=[data['timestamp']],
                y=[data['rsi']],
                name='RSI'
            ),
            row=3, col=1
        )
        
        # Update signals
        self.fig.add_trace(
            go.Scatter(
                x=[data['timestamp']],
                y=[data['signal_strength']],
                name='Signal Strength'
            ),
            row=4, col=1
        )
        
        self.fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=800
        )
        self.fig.show()

class PerformanceMetrics:
    """Class to track and analyze trading performance metrics."""
    
    def __init__(self):
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'total_loss': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'risk_reward_ratio': 0.0
        }
        self.trade_history = []
        self.equity_curve = []
    
    def update(self, trade: Dict[str, Any]) -> None:
        """Update metrics with new trade data."""
        self.trade_history.append(trade)
        self.metrics['total_trades'] += 1
        
        if trade['profit'] > 0:
            self.metrics['winning_trades'] += 1
            self.metrics['total_profit'] += trade['profit']
        else:
            self.metrics['losing_trades'] += 1
            self.metrics['total_loss'] += abs(trade['profit'])
        
        self._calculate_metrics()
    
    def _calculate_metrics(self) -> None:
        """Calculate all performance metrics."""
        total_trades = self.metrics['total_trades']
        if total_trades == 0:
            return
        
        self.metrics['win_rate'] = self.metrics['winning_trades'] / total_trades
        self.metrics['profit_factor'] = self.metrics['total_profit'] / self.metrics['total_loss'] if self.metrics['total_loss'] > 0 else float('inf')
        
        if self.metrics['winning_trades'] > 0:
            self.metrics['average_win'] = self.metrics['total_profit'] / self.metrics['winning_trades']
        if self.metrics['losing_trades'] > 0:
            self.metrics['average_loss'] = self.metrics['total_loss'] / self.metrics['losing_trades']
        
        self.metrics['risk_reward_ratio'] = self.metrics['average_win'] / self.metrics['average_loss'] if self.metrics['average_loss'] > 0 else float('inf')
        
        # Calculate equity curve metrics
        if self.equity_curve:
            returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
            self.metrics['sharpe_ratio'] = np.mean(returns) / np.std(returns) * np.sqrt(252)
            self.metrics['sortino_ratio'] = np.mean(returns) / np.std(returns[returns < 0]) * np.sqrt(252)
            self.metrics['max_drawdown'] = self._calculate_max_drawdown()
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from equity curve."""
        peak = self.equity_curve[0]
        max_drawdown = 0.0
        
        for value in self.equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return max_drawdown

class FeatureEngineer:
    """Enhanced feature engineering class with additional indicators."""
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_cache = {}
    
    @lru_cache(maxsize=1000)
    def calculate_technical_indicators(self, prices: Tuple[float], volume: Tuple[float]) -> Dict[str, float]:
        """Calculate technical indicators with caching."""
        prices = np.array(prices)
        volume = np.array(volume)
        
        # Trend indicators
        adx = ADXIndicator(
            high=pd.Series(prices),
            low=pd.Series(prices),
            close=pd.Series(prices)
        ).adx().iloc[-1]
        
        ichimoku = IchimokuIndicator(
            high=pd.Series(prices),
            low=pd.Series(prices)
        )
        
        # Momentum indicators
        stoch = StochasticOscillator(
            high=pd.Series(prices),
            low=pd.Series(prices),
            close=pd.Series(prices)
        )
        
        roc = ROCIndicator(close=pd.Series(prices)).roc()
        
        # Volume indicators
        obv = OnBalanceVolume(
            close=pd.Series(prices),
            volume=pd.Series(volume)
        ).on_balance_volume()
        
        cmf = ChaikinMoneyFlow(
            high=pd.Series(prices),
            low=pd.Series(prices),
            close=pd.Series(prices),
            volume=pd.Series(volume)
        ).chaikin_money_flow()
        
        return {
            'adx': adx,
            'ichimoku_tenkan': ichimoku.ichimoku_conversion_line().iloc[-1],
            'ichimoku_kijun': ichimoku.ichimoku_base_line().iloc[-1],
            'stoch_k': stoch.stoch().iloc[-1],
            'stoch_d': stoch.stoch_signal().iloc[-1],
            'roc': roc.iloc[-1],
            'obv': obv.iloc[-1],
            'cmf': cmf.iloc[-1]
        }
    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features with caching and parallel processing."""
        # Calculate price features
        prices = data['price'].values
        volume = data['volume'].values
        
        momentum, volatility, returns = calculate_price_features(prices)
        
        # Add basic features
        features = pd.DataFrame({
            'price_change': returns,
            'volume_change': np.diff(volume, prepend=volume[0]) / volume[0],
            'momentum': momentum,
            'volatility': volatility
        })
        
        # Add technical indicators
        tech_indicators = self.calculate_technical_indicators(
            tuple(prices[-20:]), tuple(volume[-20:])
        )
        for name, value in tech_indicators.items():
            features[name] = value
        
        # Add moving averages
        for window in [10, 20, 50, 200]:
            features[f'ma_{window}'] = SMAIndicator(
                close=pd.Series(prices),
                window=window
            ).sma_indicator()
        
        # Scale features
        features = pd.DataFrame(
            self.scaler.fit_transform(features),
            columns=features.columns
        )
        
        return features

class TradingModel:
    """Enhanced trading model with real-time capabilities."""
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.signal_analyzer = TradingSignal()
        self.visualizer = RealTimeVisualizer()
        self.performance_metrics = PerformanceMetrics()
        
        # Initialize models with optimized parameters
        self.models = {
            'rf': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                subsample=0.8,
                random_state=42
            ),
            'lr': LogisticRegression(
                penalty='l2',
                C=1.0,
                solver='liblinear',
                random_state=42
            )
        }
        
        self.ensemble = VotingClassifier(
            estimators=[(name, model) for name, model in self.models.items()],
            voting='soft',
            n_jobs=-1
        )
        
        self.feature_selector = SelectFromModel(
            estimator=RandomForestClassifier(n_estimators=100, random_state=42)
        )
    
    def start_real_time_analysis(self):
        """Start real-time analysis and visualization."""
        self.visualizer.start()
    
    def stop_real_time_analysis(self):
        """Stop real-time analysis and visualization."""
        self.visualizer.stop()
    
    def process_real_time_data(self, data: Dict[str, Any]):
        """Process real-time data and update visualizations."""
        # Update signal analysis
        self.signal_analyzer.update(data)
        
        # Update visualization
        self.visualizer.data_queue.put(data)
        
        # Make prediction
        features = self.feature_engineer.engineer_features(
            pd.DataFrame([data])
        )
        features = self.feature_selector.transform(features)
        
        prediction, probability = self.ensemble.predict(features), self.ensemble.predict_proba(features)
        
        return {
            'prediction': prediction[0],
            'probability': probability[0],
            'signal': self.signal_analyzer.signals,
            'confidence': self.signal_analyzer.confidence
        }

def save_model(model: TradingModel, filepath: str) -> None:
    """Save model with compression for reduced file size."""
    joblib.dump(model, filepath, compress=3)
    logging.info(f"Model saved to {filepath}")

def load_model(filepath: str) -> TradingModel:
    """Load model with memory mapping for large models."""
    return joblib.load(filepath, mmap_mode='r')

def main():
    """Main function with improved error handling and logging."""
    try:
        # Load data
        data = pd.read_csv('data/historical_data.csv')
        
        # Initialize model
        model = TradingModel()
        
        # Train model
        scores = model.train(data)
        logging.info(f"Training completed with mean F1: {scores['mean_f1']:.4f} ± {scores['std_f1']:.4f}")
        
        # Save model
        save_model(model, 'src/optimized_pump_dump_model.pkl')
        
    except Exception as e:
        logging.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()