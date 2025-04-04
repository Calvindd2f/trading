import pandas as pd
import numpy as np
from typing import Dict, Optional
from .base_strategy import BaseStrategy

class BollingerBandsStrategy(BaseStrategy):
    def __init__(
        self,
        period: int = 20,
        std_dev: float = 2.0,
        params: Optional[Dict] = None
    ):
        """
        Initialize Bollinger Bands Strategy.
        
        Args:
            period: Moving average period
            std_dev: Number of standard deviations for bands
            params: Additional strategy parameters
        """
        super().__init__("Bollinger Bands", params)
        self.period = period
        self.std_dev = std_dev
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals using Bollinger Bands.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            DataFrame with signals
        """
        # Calculate Bollinger Bands
        data['ma'] = data['Close'].rolling(window=self.period).mean()
        data['std'] = data['Close'].rolling(window=self.period).std()
        data['upper_band'] = data['ma'] + (self.std_dev * data['std'])
        data['lower_band'] = data['ma'] - (self.std_dev * data['std'])
        
        # Generate signals
        data['signal'] = 0
        
        # Buy signal: Price crosses below lower band
        data.loc[data['Close'] < data['lower_band'], 'signal'] = 1
        
        # Sell signal: Price crosses above upper band
        data.loc[data['Close'] > data['upper_band'], 'signal'] = -1
        
        # Remove signals during the warmup period
        data.loc[:self.period, 'signal'] = 0
        
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
        period_range: range = range(10, 51, 10),
        std_dev_range: np.ndarray = np.arange(1.5, 3.1, 0.5)
    ) -> Dict:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            data: Historical market data
            period_range: Range of periods to test
            std_dev_range: Range of standard deviations to test
            
        Returns:
            Dictionary with optimal parameters
        """
        best_sharpe = -np.inf
        best_params = {}
        
        for period in period_range:
            for std_dev in std_dev_range:
                # Test parameters
                self.period = period
                self.std_dev = std_dev
                
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
                        'period': period,
                        'std_dev': std_dev,
                        'sharpe_ratio': sharpe_ratio,
                        'total_return': total_return
                    }
        
        return best_params 