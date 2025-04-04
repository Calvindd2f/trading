import numpy as np
import pandas as pd
from typing import Dict, Optional, List
from datetime import datetime

class RiskManager:
    def __init__(
        self,
        max_position_size: float = 0.1,
        max_drawdown: float = 0.2,
        stop_loss_pct: float = 0.02,
        take_profit_pct: float = 0.04,
        max_leverage: float = 1.0
    ):
        """
        Initialize risk manager.
        
        Args:
            max_position_size: Maximum position size as percentage of portfolio
            max_drawdown: Maximum allowed drawdown
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            max_leverage: Maximum allowed leverage
        """
        self.max_position_size = max_position_size
        self.max_drawdown = max_drawdown
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_leverage = max_leverage
        self.positions = {}
        self.trades = []
        
    def calculate_position_size(
        self,
        price: float,
        account_value: float,
        volatility: float,
        confidence: float = 1.0
    ) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            price: Current price
            account_value: Account value
            volatility: Asset volatility
            confidence: Trade confidence (0-1)
            
        Returns:
            Position size in units
        """
        # Base position size
        base_size = account_value * self.max_position_size
        
        # Adjust for volatility
        vol_adjustment = 1 / (1 + volatility)
        
        # Adjust for confidence
        confidence_adjustment = confidence
        
        # Calculate final position size
        position_size = base_size * vol_adjustment * confidence_adjustment
        
        # Convert to units
        units = position_size / price
        
        return units
        
    def check_stop_loss(
        self,
        symbol: str,
        current_price: float,
        entry_price: float,
        position_type: str
    ) -> bool:
        """
        Check if stop loss is triggered.
        
        Args:
            symbol: Trading symbol
            current_price: Current price
            entry_price: Entry price
            position_type: Position type (long/short)
            
        Returns:
            True if stop loss is triggered, False otherwise
        """
        if position_type == "long":
            return current_price <= entry_price * (1 - self.stop_loss_pct)
        else:
            return current_price >= entry_price * (1 + self.stop_loss_pct)
            
    def check_take_profit(
        self,
        symbol: str,
        current_price: float,
        entry_price: float,
        position_type: str
    ) -> bool:
        """
        Check if take profit is triggered.
        
        Args:
            symbol: Trading symbol
            current_price: Current price
            entry_price: Entry price
            position_type: Position type (long/short)
            
        Returns:
            True if take profit is triggered, False otherwise
        """
        if position_type == "long":
            return current_price >= entry_price * (1 + self.take_profit_pct)
        else:
            return current_price <= entry_price * (1 - self.take_profit_pct)
            
    def calculate_drawdown(self, equity_curve: pd.Series) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            equity_curve: Series of account values
            
        Returns:
            Maximum drawdown as percentage
        """
        rolling_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        return drawdowns.min()
        
    def check_drawdown_limit(self, equity_curve: pd.Series) -> bool:
        """
        Check if drawdown limit is exceeded.
        
        Args:
            equity_curve: Series of account values
            
        Returns:
            True if drawdown limit is exceeded, False otherwise
        """
        current_drawdown = self.calculate_drawdown(equity_curve)
        return current_drawdown <= -self.max_drawdown
        
    def update_position(
        self,
        symbol: str,
        position_type: str,
        size: float,
        price: float,
        timestamp: datetime
    ) -> None:
        """
        Update position information.
        
        Args:
            symbol: Trading symbol
            position_type: Position type (long/short)
            size: Position size
            price: Entry price
            timestamp: Trade timestamp
        """
        self.positions[symbol] = {
            "type": position_type,
            "size": size,
            "entry_price": price,
            "entry_time": timestamp
        }
        
    def record_trade(
        self,
        symbol: str,
        action: str,
        size: float,
        price: float,
        timestamp: datetime
    ) -> None:
        """
        Record trade information.
        
        Args:
            symbol: Trading symbol
            action: Trade action (buy/sell)
            size: Trade size
            price: Trade price
            timestamp: Trade timestamp
        """
        self.trades.append({
            "symbol": symbol,
            "action": action,
            "size": size,
            "price": price,
            "timestamp": timestamp
        })
        
    def get_position_risk(
        self,
        symbol: str,
        current_price: float
    ) -> Dict:
        """
        Calculate current position risk metrics.
        
        Args:
            symbol: Trading symbol
            current_price: Current price
            
        Returns:
            Dictionary with risk metrics
        """
        if symbol not in self.positions:
            return {}
            
        position = self.positions[symbol]
        entry_price = position["entry_price"]
        
        if position["type"] == "long":
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price
            
        return {
            "symbol": symbol,
            "position_type": position["type"],
            "size": position["size"],
            "entry_price": entry_price,
            "current_price": current_price,
            "pnl_pct": pnl_pct,
            "stop_loss": entry_price * (1 - self.stop_loss_pct) if position["type"] == "long" else entry_price * (1 + self.stop_loss_pct),
            "take_profit": entry_price * (1 + self.take_profit_pct) if position["type"] == "long" else entry_price * (1 - self.take_profit_pct)
        } 