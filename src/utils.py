import sqlite3
import numpy as np
from datetime import datetime
from numba import jit, float64, int32, boolean
from functools import lru_cache
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def calculate_max_drawdown(equity_curve: np.ndarray) -> float:
    """Calculate the maximum drawdown of an equity curve."""
    peak = equity_curve[0]
    max_drawdown = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return max_drawdown

def calculate_sharpe_ratio(equity_curve: np.ndarray, risk_free_rate: float = 0.01) -> float:
    """Calculate the Sharpe ratio of an equity curve."""
    returns = np.diff(equity_curve) / equity_curve[:-1]
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate assuming 252 trading days in a year
    try:
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
    except ZeroDivisionError:
        sharpe_ratio = np.nan
    return sharpe_ratio * np.sqrt(252)  # Annualize the Sharpe ratio

def log_performance_metrics(
    total_trades: int,
    total_profit_loss: float,
    max_drawdown: float,
    sharpe_ratio: float,
) -> None:
    """Log performance metrics to the database."""
    try:
        conn = sqlite3.connect("trading_bot.db")
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO performance_metrics (timestamp, total_trades, total_profit_loss, max_drawdown, sharpe_ratio)
            VALUES (?, ?, ?, ?, ?)
            """,
            (datetime.now().isoformat(), total_trades, total_profit_loss, max_drawdown, sharpe_ratio),
        )
        conn.commit()
    except sqlite3.Error as e:
        logging.error(f"Error while logging performance metrics: {e}")
    finally:
        conn.close()

def calculate_position_size(account_balance: float, risk_per_trade: float) -> float:
    """Calculate the position size based on account balance and risk per trade."""
    position_size = account_balance * risk_per_trade
    return min(position_size, 10000)  # Example: MAX_POSITION_SIZE = 10000

def calculate_rsi(series: np.ndarray, window: int = 14) -> np.ndarray:
    """Calculate the RSI of a series."""
    delta = np.diff(series)
    gain = np.where(delta > 0, delta, 0).sum() / window
    loss = -np.where(delta < 0, delta, 0).sum() / window
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series: np.ndarray, slow: int = 26, fast: int = 12, signal: int = 9) -> np.ndarray:
    """Calculate the MACD of a series."""
    exp1 = np.zeros_like(series)
    exp2 = np.zeros_like(series)
    macd = np.zeros_like(series)
    signal_line = np.zeros_like(series)

    exp1[:] = series[:]
    exp2[:] = series[:]
    for i in range(1, len(series)):
        exp1[i] = exp1[i - 1] * (1 - 2 / (fast + 1)) + series[i] * (2 / (fast + 1))
        exp2[i] = exp2[i - 1] * (1 - 2 / (slow + 1)) + series[i] * (2 / (slow + 1))
        macd[i] = exp1[i] - exp2[i]
        signal_line[i] = signal_line[i - 1] * (1 - 2 / (signal + 1)) + macd[i] * (2 / (signal + 1))
    return macd - signal_line

def log_performance_metrics(
    trade_count: int,
    total_loss: float,
    max_drawdown: float,
    sharpe_ratio: float,
) -> None:
    """Log performance metrics to the console."""
    logging.info(f"Trade Count: {trade_count}")
    logging.info(f"Total Loss: {total_loss}")
    logging.info(f"Max Drawdown: {max_drawdown}")
    logging.info(f"Sharpe Ratio: {sharpe_ratio}")

def execute_trade(
    side: str,
    amount: float,
    trade_price: float,
    stop_loss_price: float,
    account_balance: float,
    risk_per_trade: float,
    equity_curve: np.ndarray,
    historical_data,
    backoff_duration: int,
) -> None:
    """Execute a trade and log its results."""
    position_size = calculate_position_size(account_balance, risk_per_trade)
    trade_amount = position_size / trade_price

    try:
        trade_loss = 0  # Simplified example without calculating actual profit/loss
    except Exception as e:
        logging.error(f"Error while executing trade: {e}")
        return

    trade_count = 1
    total_loss = trade_loss

    equity_curve.append(equity_curve[-1] - trade_loss)

    max_drawdown = calculate_max_drawdown(np.array(equity_curve))
    sharpe_ratio = calculate_sharpe_ratio(np.array(equity_curve))
    log_performance_metrics(trade_count, total_loss, max_drawdown, sharpe_ratio)

    # Adjust backoff logic based on strategy optimization
    backoff_duration = max(1, backoff_duration * 0.9 if side == "buy" else backoff_duration * 1.1)
