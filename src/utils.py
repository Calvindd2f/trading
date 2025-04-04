import logging
import sqlite3
from datetime import datetime
from functools import lru_cache
from typing import Optional, Tuple, List
import numpy as np
from numba import boolean, float64, int32, jit

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

class TradingMetrics:
    """A class to handle trading-related calculations and metrics."""
    
    @staticmethod
    @jit(float64[:])
    def calculate_max_drawdown(equity: np.ndarray) -> float:
        """
        Calculate the maximum drawdown in the equity curve.

        Parameters
        ----------
        equity : numpy.ndarray
            The equity curve to calculate the maximum drawdown from.

        Returns
        -------
        float
            The maximum drawdown in the equity curve.
        """
        if equity is None or len(equity) == 0:
            raise ValueError("Equity curve cannot be None or empty")
            
        max_drawdown = 0.0
        peak = equity[0]
        for value in equity:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
        return max_drawdown

    @staticmethod
    @jit(float64[:], float64, int32)
    def calculate_sharpe_ratio(equity_curve: np.ndarray, risk_free_rate: float = 0.01) -> float:
        """
        Calculate the Sharpe ratio for a given equity curve.

        Parameters
        ----------
        equity_curve : numpy.ndarray
            The equity curve to calculate the Sharpe ratio from.
        risk_free_rate : float, optional
            The risk-free rate to use in the calculation (default: 0.01).

        Returns
        -------
        float
            The Sharpe ratio.
        """
        if equity_curve is None or len(equity_curve) < 2:
            raise ValueError("Equity curve must have at least 2 points")
            
        returns = np.diff(equity_curve) / equity_curve[:-1]
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        mean = np.nanmean(excess_returns)
        std = np.nanstd(excess_returns)
        return np.sqrt(252) * mean / std if std != 0 else 0.0

    @staticmethod
    @lru_cache(maxsize=128)
    def calculate_position_size(account_balance: float, risk_per_trade: float, max_position: float = 10000.0) -> float:
        """
        Calculate the position size based on account balance and risk per trade.

        Parameters
        ----------
        account_balance : float
            The current account balance.
        risk_per_trade : float
            The risk percentage per trade.
        max_position : float, optional
            The maximum position size allowed (default: 10000.0).

        Returns
        -------
        float
            The calculated position size.
        """
        if account_balance <= 0 or risk_per_trade <= 0:
            raise ValueError("Account balance and risk per trade must be positive")
        return min(account_balance * risk_per_trade, max_position)

    @staticmethod
    @jit(float64[:], int32)
    def calculate_rsi(prices: np.ndarray, window: int = 14) -> np.ndarray:
        """
        Calculate the Relative Strength Index (RSI) for a given array of prices.

        Parameters
        ----------
        prices : numpy.ndarray
            Array of prices.
        window : int, optional
            Window size for RSI calculation (default: 14).

        Returns
        -------
        numpy.ndarray
            Array of RSI values.
        """
        if prices is None or len(prices) < window:
            raise ValueError("Prices array must be longer than the window size")
            
        delta = np.diff(prices)
        up, down = delta.copy(), delta.copy()
        up[up < 0] = 0
        down[down > 0] = 0

        with np.errstate('ignore'):
            roll_up = np.nan_to_num(up.rolling(window).mean())
            roll_down = np.nan_to_num(down.rolling(window).mean().abs())

        rs = roll_up / roll_down
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    @jit(float64[:], int32, int32, int32)
    def calculate_macd(series: np.ndarray, slow: int = 26, fast: int = 12, signal: int = 9) -> np.ndarray:
        """
        Calculate the Moving Average Convergence Divergence (MACD).

        Parameters
        ----------
        series : numpy.ndarray
            Array of prices.
        slow : int, optional
            Window size for slow MACD calculation (default: 26).
        fast : int, optional
            Window size for fast MACD calculation (default: 12).
        signal : int, optional
            Window size for signal line calculation (default: 9).

        Returns
        -------
        numpy.ndarray
            Array of MACD values.
        """
        if series is None or len(series) < max(slow, fast, signal):
            raise ValueError("Series must be longer than the largest window size")
            
        exp1 = series.copy()
        exp2 = series.copy()
        macd = np.zeros_like(series)
        signal_line = np.zeros_like(series)

        for i in range(1, len(series)):
            exp1[i] = exp1[i-1] * (1 - 2 / (fast + 1)) + series[i] * (2 / (fast + 1))
            exp2[i] = exp2[i-1] * (1 - 2 / (slow + 1)) + series[i] * (2 / (slow + 1))
            macd[i] = exp1[i] - exp2[i]
            signal_line[i] = signal_line[i-1] * (1 - 2 / (signal + 1)) + macd[i] * (2 / (signal + 1))
        return macd - signal_line

def log_performance_metrics(total_trades: int, total_profit_loss: float, max_drawdown: float, sharpe_ratio: float) -> None:
    """
    Log performance metrics to the database.

    Parameters
    ----------
    total_trades : int
        Total number of trades executed.
    total_profit_loss : float
        Total profit/loss from trading.
    max_drawdown : float
        Maximum drawdown experienced.
    sharpe_ratio : float
        Sharpe ratio of the trading strategy.
    """
    try:
        conn = sqlite3.connect('trading_bot.db')
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO performance_metrics 
                        (timestamp, total_trades, total_profit_loss, max_drawdown, sharpe_ratio) 
                        VALUES (?, ?, ?, ?, ?)''',
                      (datetime.now().isoformat(), total_trades, total_profit_loss, max_drawdown, sharpe_ratio))
        conn.commit()
    except sqlite3.Error as e:
        logging.error(f"Error while logging performance metrics: {e}")
    finally:
        conn.close()

def execute_trade(side):
    global trade_count, total_loss, equity_curve, historical_data, backoff_duration

    if historical_data is None:
        logging.error(
            "Error while fetching historical data. Historical data is None.")
        return

    try:
        trade_price = historical_data.iloc[-1]['price']
    except IndexError:
        logging.error("Error while fetching historical data.")
        return

    position_size = TradingMetrics.calculate_position_size(
        10000, 0.01)  # Example: 1% risk per trade
    trade_amount = position_size / trade_price

    if side == 'buy':
        # Example: STOP_LOSS_PERCENT = 0.05
        stop_loss_price = trade_price * (1 - 0.05)
    else:
        stop_loss_price = trade_price * (1 + 0.05)

    try:
        # Simulate trade execution (use a dummy price for simplicity)
        trade_loss, = 0  # Simplified example without calculating actual profit/loss
    except Exception as e:
        logging.error("Error while executing trade: %s", e)
        return

    trade_count += 1
    total_loss += trade_loss

    if equity_curve is None:
        equity_curve = [10000 - trade_loss]  # Starting equity of 10000
    else:
        equity_curve[-1] -= trade_loss

    logging.info("Executed %s trade for %s at price %s, stop loss at %s", side, trade_amount, trade_price, stop_loss_price)

    max_drawdown = TradingMetrics.calculate_max_drawdown(np.array(equity_curve))
    sharpe_ratio = TradingMetrics.calculate_sharpe_ratio(np.array(equity_curve))
    log_performance_metrics(trade_count, total_loss,
                            max_drawdown, sharpe_ratio)

    # Adjust backoff logic based on strategy optimization
    backoff_duration = max(1, backoff_duration *
                           0.9 if side == 'buy' else backoff_duration * 1.1)
