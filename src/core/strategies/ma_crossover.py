import pandas as pd
import numpy as np
from typing import Dict, Optional
from .base_strategy import BaseStrategy

class MACrossoverStrategy(BaseStrategy):
    def __init__(
        self,
        fast_period: int = 10,
        slow_period: int = 20,
        params: Optional[Dict] = None
    ):
        """
        Initialize Moving Average Crossover Strategy.
        
        Args:
            fast_period: Fast moving average period
            slow_period: Slow moving average period
            params: Additional strategy parameters
        """
        super().__init__("MA Crossover", params)
        self.fast_period = fast_period
        self.slow_period = slow_period
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals using moving average crossover.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            DataFrame with signals
        """
        # Calculate moving averages
        data['fast_ma'] = data['Close'].rolling(window=self.fast_period).mean()
        data['slow_ma'] = data['Close'].rolling(window=self.slow_period).mean()
        
        # Generate signals
        data['signal'] = 0
        
        # Bullish crossover (fast MA crosses above slow MA)
        data.loc[data['fast_ma'] > data['slow_ma'], 'signal'] = 1
        
        # Bearish crossover (fast MA crosses below slow MA)
        data.loc[data['fast_ma'] < data['slow_ma'], 'signal'] = -1
        
        # Remove signals during the warmup period
        data.loc[:self.slow_period, 'signal'] = 0
        
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
        fast_range: range = range(5, 21, 5),
        slow_range: range = range(20, 51, 10)
    ) -> Dict:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            data: Historical market data
            fast_range: Range of fast MA periods to test
            slow_range: Range of slow MA periods to test
            
        Returns:
            Dictionary with optimal parameters
        """
        best_sharpe = -np.inf
        best_params = {}
        
        for fast in fast_range:
            for slow in slow_range:
                if fast >= slow:
                    continue
                    
                # Test parameters
                self.fast_period = fast
                self.slow_period = slow
                
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
                        'fast_period': fast,
                        'slow_period': slow,
                        'sharpe_ratio': sharpe_ratio,
                        'total_return': total_return
                    }
        
        return best_params 