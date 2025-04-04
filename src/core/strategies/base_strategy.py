from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Optional, Tuple
import numpy as np
from datetime import datetime

class BaseStrategy(ABC):
    def __init__(self, name: str, params: Optional[Dict] = None):
        """
        Initialize the base strategy.
        
        Args:
            name: Strategy name
            params: Strategy parameters
        """
        self.name = name
        self.params = params or {}
        self.positions = {}
        self.trades = []
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from market data.
        
        Args:
            data: DataFrame with market data
            
        Returns:
            DataFrame with signals
        """
        pass
        
    def calculate_position_size(
        self,
        price: float,
        account_value: float,
        risk_per_trade: float = 0.02
    ) -> float:
        """
        Calculate position size based on risk management rules.
        
        Args:
            price: Current price
            account_value: Total account value
            risk_per_trade: Maximum risk per trade as percentage
            
        Returns:
            Position size in units
        """
        risk_amount = account_value * risk_per_trade
        position_size = risk_amount / price
        return position_size
        
    def execute_trade(
        self,
        symbol: str,
        signal: int,
        price: float,
        account_value: float,
        timestamp: datetime
    ) -> Optional[Dict]:
        """
        Execute a trade based on signal.
        
        Args:
            symbol: Trading symbol
            signal: Trading signal (-1, 0, 1)
            price: Current price
            account_value: Account value
            timestamp: Trade timestamp
            
        Returns:
            Trade details if executed, None otherwise
        """
        if signal == 0:
            return None
            
        position_size = self.calculate_position_size(price, account_value)
        
        # Check if we already have a position
        current_position = self.positions.get(symbol, 0)
        
        # Determine trade action
        if signal == 1 and current_position <= 0:
            # Buy signal and no position or short position
            trade = {
                "symbol": symbol,
                "action": "buy",
                "size": position_size,
                "price": price,
                "timestamp": timestamp
            }
            self.positions[symbol] = position_size
            self.trades.append(trade)
            return trade
            
        elif signal == -1 and current_position >= 0:
            # Sell signal and no position or long position
            trade = {
                "symbol": symbol,
                "action": "sell",
                "size": position_size,
                "price": price,
                "timestamp": timestamp
            }
            self.positions[symbol] = -position_size
            self.trades.append(trade)
            return trade
            
        return None
        
    def calculate_returns(self, trades: list) -> Tuple[float, float]:
        """
        Calculate strategy returns and Sharpe ratio.
        
        Args:
            trades: List of executed trades
            
        Returns:
            Tuple of (total_return, sharpe_ratio)
        """
        if not trades:
            return 0.0, 0.0
            
        returns = []
        for i in range(1, len(trades)):
            prev_trade = trades[i-1]
            curr_trade = trades[i]
            
            if prev_trade["action"] == "buy" and curr_trade["action"] == "sell":
                ret = (curr_trade["price"] - prev_trade["price"]) / prev_trade["price"]
                returns.append(ret)
                
        returns = np.array(returns)
        total_return = np.prod(1 + returns) - 1
        sharpe_ratio = np.mean(returns) / np.std(returns) if len(returns) > 1 else 0
        
        return total_return, sharpe_ratio 