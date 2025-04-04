import pandas as pd
import numpy as np
from typing import Dict, Optional
from .base_strategy import BaseStrategy

class MeanReversionStrategy(BaseStrategy):
    def __init__(
        self,
        lookback_period: int = 20,
        entry_threshold: float = 2.0,
        exit_threshold: float = 0.5,
        params: Optional[Dict] = None
    ):
        """
        Initialize Mean Reversion Strategy.
        
        Args:
            lookback_period: Period for calculating mean and standard deviation
            entry_threshold: Number of standard deviations for entry
            exit_threshold: Number of standard deviations for exit
            params: Additional strategy parameters
        """
        super().__init__("Mean Reversion", params)
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals using Mean Reversion.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            DataFrame with signals
        """
        # Calculate rolling mean and standard deviation
        data['mean'] = data['Close'].rolling(window=self.lookback_period).mean()
        data['std'] = data['Close'].rolling(window=self.lookback_period).std()
        
        # Calculate z-score
        data['z_score'] = (data['Close'] - data['mean']) / data['std']
        
        # Generate signals
        data['signal'] = 0
        
        # Buy signal: Price is significantly below mean
        data.loc[data['z_score'] < -self.entry_threshold, 'signal'] = 1
        
        # Sell signal: Price is significantly above mean
        data.loc[data['z_score'] > self.entry_threshold, 'signal'] = -1
        
        # Exit long positions when price returns to mean
        data.loc[(data['z_score'] > -self.exit_threshold) & (data['signal'].shift(1) == 1), 'signal'] = 0
        
        # Exit short positions when price returns to mean
        data.loc[(data['z_score'] < self.exit_threshold) & (data['signal'].shift(1) == -1), 'signal'] = 0
        
        # Remove signals during the warmup period
        data.loc[:self.lookback_period, 'signal'] = 0
        
        # Add position sizing
        data['position_size'] = data.apply(
            lambda row: self.calculate_position_size(
                row['Close'],
                self.params.get('account_value', 100000),
                self.params.get('risk_per_trade', 0.02)
            ) if row['signal'] != 0 else 0,
            axis=1
        )
        
        return data
        
    def optimize_parameters(
        self,
        data: pd.DataFrame,
        lookback_range: range = range(10, 51, 10),
        entry_range: np.ndarray = np.arange(1.5, 3.1, 0.5),
        exit_range: np.ndarray = np.arange(0.5, 1.6, 0.5)
    ) -> Dict:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            data: Historical market data
            lookback_range: Range of lookback periods to test
            entry_range: Range of entry thresholds to test
            exit_range: Range of exit thresholds to test
            
        Returns:
            Dictionary with optimal parameters
        """
        best_sharpe = -np.inf
        best_params = {}
        
        for lookback in lookback_range:
            for entry in entry_range:
                for exit in exit_range:
                    # Test parameters
                    self.lookback_period = lookback
                    self.entry_threshold = entry
                    self.exit_threshold = exit
                    
                    # Generate signals
                    signals = self.generate_signals(data)
                    
                    # Calculate returns
                    trades = []
                    for i in range(1, len(signals)):
                        if signals.iloc[i]['signal'] != signals.iloc[i-1]['signal']:
                            trades.append({
                                'action': 'buy' if signals.iloc[i]['signal'] == 1 else 'sell',
                                'price': signals.iloc[i]['Close'],
                                'timestamp': signals.index[i]
                            })
                    
                    # Calculate performance
                    total_return, sharpe_ratio = self.calculate_returns(trades)
                    
                    # Update best parameters
                    if sharpe_ratio > best_sharpe:
                        best_sharpe = sharpe_ratio
                        best_params = {
                            'lookback_period': lookback,
                            'entry_threshold': entry,
                            'exit_threshold': exit,
                            'sharpe_ratio': sharpe_ratio,
                            'total_return': total_return
                        }
        
        return best_params 