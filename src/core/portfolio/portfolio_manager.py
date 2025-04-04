import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from ..risk_management.risk_manager import RiskManager

class PortfolioManager:
    def __init__(
        self,
        initial_capital: float = 100000.0,
        risk_manager: Optional[RiskManager] = None
    ):
        """
        Initialize portfolio manager.
        
        Args:
            initial_capital: Initial capital
            risk_manager: Risk manager instance
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = pd.Series()
        self.risk_manager = risk_manager or RiskManager()
        
    def update_position(
        self,
        symbol: str,
        action: str,
        size: float,
        price: float,
        timestamp: datetime
    ) -> None:
        """
        Update portfolio position.
        
        Args:
            symbol: Trading symbol
            action: Trade action (buy/sell)
            size: Position size
            price: Trade price
            timestamp: Trade timestamp
        """
        # Record trade
        self.trades.append({
            "symbol": symbol,
            "action": action,
            "size": size,
            "price": price,
            "timestamp": timestamp
        })
        
        # Update position
        if action == "buy":
            if symbol in self.positions:
                self.positions[symbol]["size"] += size
                self.positions[symbol]["avg_price"] = (
                    self.positions[symbol]["avg_price"] * self.positions[symbol]["size"] +
                    price * size
                ) / (self.positions[symbol]["size"] + size)
            else:
                self.positions[symbol] = {
                    "size": size,
                    "avg_price": price,
                    "entry_time": timestamp
                }
        else:  # sell
            if symbol in self.positions:
                self.positions[symbol]["size"] -= size
                if self.positions[symbol]["size"] <= 0:
                    del self.positions[symbol]
                    
        # Update risk manager
        self.risk_manager.update_position(
            symbol,
            "long" if action == "buy" else "short",
            size,
            price,
            timestamp
        )
        
    def calculate_position_value(self, symbol: str, current_price: float) -> float:
        """
        Calculate current position value.
        
        Args:
            symbol: Trading symbol
            current_price: Current price
            
        Returns:
            Position value
        """
        if symbol not in self.positions:
            return 0.0
            
        position = self.positions[symbol]
        return position["size"] * current_price
        
    def calculate_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate total portfolio value.
        
        Args:
            current_prices: Dictionary of current prices
            
        Returns:
            Total portfolio value
        """
        total_value = self.current_capital
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                total_value += self.calculate_position_value(symbol, current_prices[symbol])
                
        return total_value
        
    def update_equity_curve(self, timestamp: datetime, portfolio_value: float) -> None:
        """
        Update equity curve.
        
        Args:
            timestamp: Current timestamp
            portfolio_value: Current portfolio value
        """
        self.equity_curve[timestamp] = portfolio_value
        
    def calculate_returns(self) -> Dict:
        """
        Calculate portfolio performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        if len(self.equity_curve) < 2:
            return {}
            
        returns = self.equity_curve.pct_change().dropna()
        
        total_return = (self.equity_curve.iloc[-1] / self.initial_capital) - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        max_drawdown = self.risk_manager.calculate_drawdown(self.equity_curve)
        
        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "num_trades": len(self.trades),
            "win_rate": self._calculate_win_rate()
        }
        
    def _calculate_win_rate(self) -> float:
        """
        Calculate win rate of trades.
        
        Returns:
            Win rate as percentage
        """
        if not self.trades:
            return 0.0
            
        winning_trades = 0
        for i in range(1, len(self.trades), 2):
            if i < len(self.trades):
                entry_trade = self.trades[i-1]
                exit_trade = self.trades[i]
                
                if entry_trade["action"] == "buy":
                    if exit_trade["price"] > entry_trade["price"]:
                        winning_trades += 1
                else:
                    if exit_trade["price"] < entry_trade["price"]:
                        winning_trades += 1
                        
        return winning_trades / (len(self.trades) // 2)
        
    def get_position_summary(self, current_prices: Dict[str, float]) -> pd.DataFrame:
        """
        Get summary of all positions.
        
        Args:
            current_prices: Dictionary of current prices
            
        Returns:
            DataFrame with position summary
        """
        positions_data = []
        
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                position_value = self.calculate_position_value(symbol, current_price)
                pnl = position_value - (position["size"] * position["avg_price"])
                pnl_pct = pnl / (position["size"] * position["avg_price"])
                
                positions_data.append({
                    "symbol": symbol,
                    "size": position["size"],
                    "avg_price": position["avg_price"],
                    "current_price": current_price,
                    "value": position_value,
                    "pnl": pnl,
                    "pnl_pct": pnl_pct
                })
                
        return pd.DataFrame(positions_data) 