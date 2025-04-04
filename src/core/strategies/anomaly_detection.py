from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from .base_strategy import BaseStrategy
from .anomaly_methods import AnomalyDetector

class AnomalyDetectionStrategy(BaseStrategy):
    """
    Advanced Anomaly Detection Strategy that identifies unusual market patterns
    using state-of-the-art algorithms and integrates with technical analysis.
    """
    
    def __init__(
        self,
        methods: List[str] = ['autoencoder', 'isolation_forest', 'one_class_svm', 'lof', 'dbscan'],
        weights: Optional[Dict[str, float]] = None,
        anomaly_threshold: float = 0.8,
        confidence_threshold: float = 0.7,
        **kwargs
    ):
        """
        Initialize the Anomaly Detection Strategy.
        
        Args:
            methods: List of anomaly detection methods to use
            weights: Weights for each method in ensemble
            anomaly_threshold: Threshold for anomaly detection
            confidence_threshold: Minimum confidence required for signals
        """
        super().__init__(**kwargs)
        self.methods = methods
        self.weights = weights
        self.anomaly_threshold = anomaly_threshold
        self.confidence_threshold = confidence_threshold
        self.detector = AnomalyDetector()
        
    def train(self, data: pd.DataFrame) -> None:
        """
        Train the anomaly detection models.
        
        Args:
            data: Historical market data for training
        """
        # Train each specified method
        if 'autoencoder' in self.methods:
            self.detector.train_autoencoder(data)
            
        if 'isolation_forest' in self.methods:
            self.detector.train_isolation_forest(data, contamination='auto')
            
        if 'one_class_svm' in self.methods:
            self.detector.train_one_class_svm(data, gamma='auto')
            
        if 'lof' in self.methods:
            self.detector.train_local_outlier_factor(data, n_neighbors='auto')
            
        if 'dbscan' in self.methods:
            self.detector.train_dbscan(data)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on anomaly detection and market regime.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with signals
        """
        # Get anomaly scores and confidence metrics
        scores, confidence = self.detector.ensemble_detect(data, weights=self.weights)
        
        # Create signals DataFrame
        signals = pd.DataFrame(index=data.index)
        signals['anomaly_score'] = scores
        signals['signal'] = 0
        signals['confidence'] = confidence['mean_score']
        signals['confidence_std'] = confidence['std_score']
        
        # Calculate market regime features
        df = self.detector._calculate_technical_indicators(data)
        
        # Generate signals based on anomaly scores, confidence, and market regime
        for i in range(len(signals)):
            anomaly_score = scores[i]
            confidence = signals['confidence'].iloc[i]
            
            # Market regime features
            trend_strength = df['trend_strength'].iloc[i]
            volatility_regime = df['volatility_regime'].iloc[i]
            volume_regime = df['volume_regime'].iloc[i]
            
            # Technical indicators
            rsi = df['rsi'].iloc[i]
            macd_diff = df['macd_diff'].iloc[i]
            bb_pct = df['bb_pct'].iloc[i]
            stoch_k = df['stoch_k'].iloc[i]
            stoch_d = df['stoch_d'].iloc[i]
            obv = df['obv'].iloc[i]
            cmf = df['cmf'].iloc[i]
            
            # Market regime classification
            is_high_volatility = volatility_regime > df['volatility_regime'].rolling(20).mean().iloc[i]
            is_high_volume = volume_regime > df['volume_regime'].rolling(20).mean().iloc[i]
            is_strong_trend = trend_strength > 0.02
            
            # Generate signal based on anomaly, confidence, and market regime
            if confidence > self.confidence_threshold:
                if anomaly_score > self.anomaly_threshold:
                    # Strong anomaly detected
                    if (rsi < 30 and macd_diff > 0 and stoch_k < 20) or \
                       (bb_pct < 0.2 and obv > 0 and cmf > 0):
                        signals.iloc[i, signals.columns.get_loc('signal')] = 1  # Strong buy
                    elif (rsi > 70 and macd_diff < 0 and stoch_k > 80) or \
                         (bb_pct > 0.8 and obv < 0 and cmf < 0):
                        signals.iloc[i, signals.columns.get_loc('signal')] = -1  # Strong sell
                elif anomaly_score < -self.anomaly_threshold:
                    # Strong negative anomaly detected
                    if (rsi > 70 and macd_diff < 0 and stoch_k > 80) or \
                       (bb_pct > 0.8 and obv < 0 and cmf < 0):
                        signals.iloc[i, signals.columns.get_loc('signal')] = -1  # Strong sell
                    elif (rsi < 30 and macd_diff > 0 and stoch_k < 20) or \
                         (bb_pct < 0.2 and obv > 0 and cmf > 0):
                        signals.iloc[i, signals.columns.get_loc('signal')] = 1  # Strong buy
                else:
                    # Normal market conditions
                    if is_high_volatility and is_high_volume:
                        # High volatility and volume regime
                        if rsi < 30 and macd_diff > 0 and stoch_k < 20:
                            signals.iloc[i, signals.columns.get_loc('signal')] = 0.5  # Weak buy
                        elif rsi > 70 and macd_diff < 0 and stoch_k > 80:
                            signals.iloc[i, signals.columns.get_loc('signal')] = -0.5  # Weak sell
                    elif is_strong_trend:
                        # Strong trend regime
                        if macd_diff > 0 and stoch_k > stoch_d:
                            signals.iloc[i, signals.columns.get_loc('signal')] = 0.3  # Very weak buy
                        elif macd_diff < 0 and stoch_k < stoch_d:
                            signals.iloc[i, signals.columns.get_loc('signal')] = -0.3  # Very weak sell
        
        # Add additional confidence metrics
        signals['agreement_score'] = confidence['agreement_score']
        signals['consistency_score'] = confidence['consistency_score']
        signals['silhouette_score'] = confidence['silhouette_score']
        
        return signals
    
    def optimize_parameters(
        self,
        data: pd.DataFrame,
        param_grid: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Optimize strategy parameters using grid search with enhanced metrics.
        
        Args:
            data: Market data for optimization
            param_grid: Parameter grid for optimization
            
        Returns:
            Dict[str, Any]: Best parameters
        """
        if param_grid is None:
            param_grid = {
                'anomaly_threshold': [0.7, 0.8, 0.9],
                'confidence_threshold': [0.6, 0.7, 0.8],
                'weights': [
                    {'autoencoder': 0.3, 'isolation_forest': 0.2, 'one_class_svm': 0.2, 'lof': 0.2, 'dbscan': 0.1},
                    {'autoencoder': 0.4, 'isolation_forest': 0.2, 'one_class_svm': 0.2, 'lof': 0.1, 'dbscan': 0.1},
                    {'autoencoder': 0.5, 'isolation_forest': 0.2, 'one_class_svm': 0.2, 'lof': 0.1, 'dbscan': 0.0}
                ]
            }
            
        best_score = float('-inf')
        best_params = None
        
        for threshold in param_grid['anomaly_threshold']:
            for conf_threshold in param_grid['confidence_threshold']:
                for weights in param_grid['weights']:
                    self.anomaly_threshold = threshold
                    self.confidence_threshold = conf_threshold
                    self.weights = weights
                    
                    self.train(data)
                    signals = self.generate_signals(data)
                    
                    # Calculate performance score with additional metrics
                    score = self._calculate_performance_score(signals, data)
                    
                    if score > best_score:
                        best_score = score
                        best_params = {
                            'anomaly_threshold': threshold,
                            'confidence_threshold': conf_threshold,
                            'weights': weights
                        }
        
        return best_params
    
    def _calculate_performance_score(
        self,
        signals: pd.DataFrame,
        data: pd.DataFrame
    ) -> float:
        """
        Calculate enhanced performance score for parameter optimization.
        
        Args:
            signals: Generated signals
            data: Market data
            
        Returns:
            float: Performance score
        """
        # Calculate returns
        returns = data['close'].pct_change()
        
        # Calculate strategy returns
        strategy_returns = signals['signal'].shift(1) * returns
        
        # Calculate Sharpe ratio
        sharpe_ratio = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
        
        # Calculate win rate
        win_rate = len(strategy_returns[strategy_returns > 0]) / len(strategy_returns[strategy_returns != 0])
        
        # Calculate max drawdown
        cum_returns = (1 + strategy_returns).cumprod()
        max_drawdown = (cum_returns.expanding().max() - cum_returns).max()
        
        # Calculate additional metrics
        profit_factor = abs(strategy_returns[strategy_returns > 0].sum() / strategy_returns[strategy_returns < 0].sum())
        avg_trade_duration = len(signals) / len(signals[signals['signal'] != 0])
        signal_quality = signals['agreement_score'].mean() * signals['consistency_score'].mean()
        
        # Combined score with additional metrics
        score = (sharpe_ratio * (1 - max_drawdown) * win_rate * 
                profit_factor * (1 / avg_trade_duration) * signal_quality)
        
        return score 