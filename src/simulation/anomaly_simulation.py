import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, Any, List
import joblib
import os

from src.core.market_data_fetcher import MarketDataFetcher
from src.core.strategies.strategy_combiner import StrategyCombiner

class AnomalySimulation:
    """
    Simulates trading with integrated anomaly detection and technical analysis strategies.
    """
    
    def __init__(
        self,
        initial_balance: float = 10000.0,
        position_size: float = 0.1,
        stop_loss: float = 0.02,
        take_profit: float = 0.04
    ):
        """
        Initialize the simulation.
        
        Args:
            initial_balance: Initial account balance
            position_size: Position size as fraction of balance
            stop_loss: Stop loss percentage
            take_profit: Take profit percentage
        """
        self.initial_balance = initial_balance
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.balance = initial_balance
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
    def run_simulation(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1h"
    ) -> Dict[str, Any]:
        """
        Run the trading simulation.
        
        Args:
            symbol: Trading pair symbol
            start_date: Start date for simulation
            end_date: End date for simulation
            interval: Data interval
            
        Returns:
            Dict[str, Any]: Simulation results
        """
        # Load trained model
        model_path = "models/anomaly_detection_model.pkl"
        if not os.path.exists(model_path):
            raise FileNotFoundError("Trained model not found. Please train the model first.")
        
        combiner = joblib.load(model_path)
        
        # Fetch simulation data
        fetcher = MarketDataFetcher()
        data = fetcher.fetch_historical_data(
            symbol=symbol,
            start_time=start_date,
            end_time=end_date,
            interval=interval
        )
        
        # Initialize simulation state
        self.balance = self.initial_balance
        self.positions = {}
        self.trades = []
        self.equity_curve = [self.balance]
        
        # Run simulation
        for i in range(1, len(data)):
            current_data = data.iloc[:i+1]
            current_price = data.iloc[i]['close']
            
            # Generate signals
            signals = combiner.generate_signals(current_data)
            current_signal = signals.iloc[i]['signal']
            anomaly_score = signals.iloc[i].get('anomaly_score', 0)
            
            # Update positions
            self._update_positions(current_price, current_signal, anomaly_score)
            
            # Record equity
            self.equity_curve.append(self.balance)
        
        # Calculate performance metrics
        returns = pd.Series(self.equity_curve).pct_change()
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        max_drawdown = (pd.Series(self.equity_curve).expanding().max() - pd.Series(self.equity_curve)).max()
        total_return = (self.balance - self.initial_balance) / self.initial_balance
        
        results = {
            'final_balance': self.balance,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': len(self.trades),
            'win_rate': len([t for t in self.trades if t['profit'] > 0]) / len(self.trades) if self.trades else 0,
            'equity_curve': self.equity_curve,
            'trades': self.trades
        }
        
        return results
    
    def _update_positions(
        self,
        current_price: float,
        signal: float,
        anomaly_score: float
    ) -> None:
        """
        Update positions based on signals and anomaly scores.
        
        Args:
            current_price: Current market price
            signal: Trading signal
            anomaly_score: Anomaly detection score
        """
        # Calculate position size based on anomaly score
        position_multiplier = 1.0 + np.abs(anomaly_score) if np.abs(anomaly_score) > 0.8 else 1.0
        current_position_size = self.position_size * position_multiplier
        
        # Update existing positions
        for symbol, position in list(self.positions.items()):
            # Check stop loss and take profit
            price_change = (current_price - position['entry_price']) / position['entry_price']
            
            if (position['side'] == 'long' and price_change <= -self.stop_loss) or \
               (position['side'] == 'short' and price_change >= self.stop_loss):
                # Stop loss hit
                self._close_position(symbol, current_price, 'stop_loss')
            elif (position['side'] == 'long' and price_change >= self.take_profit) or \
                 (position['side'] == 'short' and price_change <= -self.take_profit):
                # Take profit hit
                self._close_position(symbol, current_price, 'take_profit')
        
        # Open new positions based on signal
        if signal > 0.5 and 'BTC/USD' not in self.positions:
            # Open long position
            self._open_position('BTC/USD', 'long', current_price, current_position_size)
        elif signal < -0.5 and 'BTC/USD' not in self.positions:
            # Open short position
            self._open_position('BTC/USD', 'short', current_price, current_position_size)
    
    def _open_position(
        self,
        symbol: str,
        side: str,
        price: float,
        size: float
    ) -> None:
        """
        Open a new position.
        
        Args:
            symbol: Trading pair symbol
            side: Position side ('long' or 'short')
            price: Entry price
            size: Position size
        """
        position_value = self.balance * size
        self.positions[symbol] = {
            'side': side,
            'entry_price': price,
            'size': size,
            'value': position_value
        }
    
    def _close_position(
        self,
        symbol: str,
        price: float,
        reason: str
    ) -> None:
        """
        Close an existing position.
        
        Args:
            symbol: Trading pair symbol
            price: Exit price
            reason: Reason for closing position
        """
        position = self.positions.pop(symbol)
        price_change = (price - position['entry_price']) / position['entry_price']
        
        if position['side'] == 'short':
            price_change = -price_change
        
        profit = position['value'] * price_change
        self.balance += profit
        
        self.trades.append({
            'symbol': symbol,
            'side': position['side'],
            'entry_price': position['entry_price'],
            'exit_price': price,
            'profit': profit,
            'reason': reason
        })
    
    def plot_results(self, results: Dict[str, Any]) -> None:
        """
        Plot simulation results.
        
        Args:
            results: Simulation results
        """
        plt.figure(figsize=(15, 10))
        
        # Plot equity curve
        plt.subplot(2, 1, 1)
        plt.plot(results['equity_curve'])
        plt.title('Equity Curve')
        plt.xlabel('Time')
        plt.ylabel('Balance')
        
        # Plot trades
        plt.subplot(2, 1, 2)
        for trade in results['trades']:
            color = 'green' if trade['profit'] > 0 else 'red'
            plt.scatter(trade['exit_price'], trade['profit'], color=color)
        plt.title('Trade Results')
        plt.xlabel('Exit Price')
        plt.ylabel('Profit/Loss')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Initialize simulation
    simulation = AnomalySimulation(
        initial_balance=10000.0,
        position_size=0.1,
        stop_loss=0.02,
        take_profit=0.04
    )
    
    # Run simulation
    print("Running simulation...")
    results = simulation.run_simulation(
        symbol="BTC/USD",
        start_date="2024-01-01",
        end_date="2024-03-31",
        interval="1h"
    )
    
    # Print results
    print("\nSimulation Results:")
    print(f"Final Balance: ${results['final_balance']:.2f}")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Number of Trades: {results['num_trades']}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    
    # Plot results
    simulation.plot_results(results) 