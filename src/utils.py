import sqlite3
import numpy as np
from datetime import datetime
from numba import jit
from functools import lru_cache


@jit(float64[:])
def calculate_max_drawdown(equity):
    """
    Calculate the maximum drawdown in the equity curve.

    Parameters
    ----------
    equity : numpy.float64[:]
        The equity curve to calculate the maximum drawdown from.

    Returns
    -------
    float
        The maximum drawdown in the equity curve.
    """
    max_drawdown = 0.0
    peak = equity[0]
    for value in equity:
        if value is None:
            logging.error(f"Error while calculating max drawdown: found null value in equity curve.")
            return max_drawdown
        if value > peak:
            peak = value
        else:
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
    return max_drawdown

@jit(float64[:], float64, int32)
def calculate_sharpe_ratio(equity_curve, risk_free_rate=0.01):
    returns = np.diff(equity_curve) / equity_curve[:-1]
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate assuming 252 trading days in a year
    mean = np.nanmean(excess_returns)
    std = np.nanstd(excess_returns)
    if std == 0:
        return 0
    return np.sqrt(252) * mean / std

def log_performance_metrics(total_trades, total_profit_loss, max_drawdown, sharpe_ratio):
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()
    cursor.execute('''INSERT INTO performance_metrics (timestamp, total_trades, total_profit_loss, max_drawdown, sharpe_ratio) VALUES (?, ?, ?, ?, ?)''',
                   (datetime.now().isoformat(), total_trades, total_profit_loss, max_drawdown, sharpe_ratio))
    conn.commit()
    conn.close()

@lru_cache(maxsize=None)
@jit(float64, float64, float64, int32)
def calculate_position_size(account_balance: float, risk_per_trade: float) -> float:
    if account_balance is None or risk_per_trade is None:
        raise ValueError("Account balance and risk per trade cannot be None.")
    return min(account_balance * risk_per_trade, float64(10000.0))

# Function to calculate RSI using Numba
def calculate_rsi(prices: np.ndarray, window: int = 14) -> np.ndarray:
    """
    Calculate the Relative Strength Index (RSI) for a given array of prices.

    Parameters:
    prices (np.ndarray): Array of prices
    window (int, optional): Window size for RSI calculation (default: 14)

    Returns:
    np.ndarray: Array of RSI values
    """
    if prices is None:
        raise ValueError("prices cannot be None")

    delta = prices.diff(1)
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    with np.errstate('ignore'):
        roll_up = np.nan_to_num(up.rolling(window).mean())
        roll_down = np.nan_to_num(down.rolling(window).mean().abs())

    rs = roll_up / roll_down
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi
# Function to calculate MACD using Numba
@jit(nopython=True)
def calculate_macd(series, slow=26, fast=12, signal=9):
    """
    Calculate the Moving Average Convergence Divergence (MACD) for a given array of prices.

    Parameters:
    series (np.ndarray): Array of prices
    slow (int, optional): Window size for slow MACD calculation (default: 26)
    fast (int, optional): Window size for fast MACD calculation (default: 12)
    signal (int, optional): Window size for signal line calculation (default: 9)

    Returns:
    np.ndarray: Array of MACD values
    """
    exp1 = np.zeros_like(series)
    exp2 = np.zeros_like(series)
    macd = np.zeros_like(series)
    signal_line = np.zeros_like(series)
    
    exp1[:] = series[:]
    exp2[:] = series[:]
    for i in range(1, len(series)):
        exp1[i] = exp1[i-1] * (1 - 2 / (fast + 1)) + series[i] * (2 / (fast + 1))
        exp2[i] = exp2[i-1] * (1 - 2 / (slow + 1)) + series[i] * (2 / (slow + 1))
        macd[i] = exp1[i] - exp2[i]
        signal_line[i] = signal_line[i-1] * (1 - 2 / (signal + 1)) + macd[i] * (2 / (signal + 1))
    return macd - signal_line

def execute_trade(side, amount):
    global trade_count, total_loss, equity_curve, historical_data, backoff_duration

    if historical_data is None:
        logging.error("Error while fetching historical data. Historical data is None.")
        return

    try:
        trade_price = historical_data.iloc[-1]['price']
    except IndexError:
        logging.error("Error while fetching historical data.")
        return

    trade_price = historical_data.iloc[-1]['price']
    position_size = calculate_position_size(10000, 0.01)  # Example: 1% risk per trade
    trade_amount = position_size / trade_price

    if side == 'buy':
        stop_loss_price = trade_price * (1 - 0.05)  # Example: STOP_LOSS_PERCENT = 0.05
    else:
        stop_loss_price = trade_price * (1 + 0.05)

    try:
        # Simulate trade execution (use a dummy price for simplicity)
        trade_loss, = 0  # Simplified example without calculating actual profit/loss
    except Exception as e:
        logging.error(f"Error while executing trade: {e}")
        return

    trade_count += 1
    total_loss += trade_loss

    if equity_curve is None:
        equity_curve = [10000 - trade_loss]  # Starting equity of 10000
    else:
        equity_curve[-1] -= trade_loss

    logging.info(f"Executed {side} trade for {trade_amount} at price {trade_price}, stop loss at {stop_loss_price}")

    max_drawdown = calculate_max_drawdown(np.array(equity_curve))
    sharpe_ratio = calculate_sharpe_ratio(np.array(equity_curve))
    log_performance_metrics(trade_count, total_loss, max_drawdown, sharpe_ratio)

    # Adjust backoff logic based on strategy optimization
    global backoff_duration
    backoff_duration = max(1, backoff_duration * 0.9 if side == 'buy' else backoff_duration * 1.1)
