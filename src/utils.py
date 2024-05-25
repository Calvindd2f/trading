import sqlite3
import numpy as np
from datetime import datetime
from numba import jit, float64, int32, boolean
from functools import lru_cache
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

@jit(float64(float64[:]), nopython=True)
def calculate_max_drawdown(equity_curve):
    peak = equity_curve[0]
    max_drawdown = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return max_drawdown

@jit(float64(float64[:], float64), nopython=True)
def calculate_sharpe_ratio(equity_curve, risk_free_rate=0.01):
    returns = np.diff(equity_curve) / equity_curve[:-1]
    excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate assuming 252 trading days in a year
    try:
        sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
    except ZeroDivisionError:
        sharpe_ratio = np.nan
    return sharpe_ratio * np.sqrt(252)  # Annualize the Sharpe ratio

def log_performance_metrics(total_trades, total_profit_loss, max_drawdown, sharpe_ratio):
    try:
        conn = sqlite3.connect('trading_bot.db')
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO performance_metrics (timestamp, total_trades, total_profit_loss, max_drawdown, sharpe_ratio) VALUES (?, ?, ?, ?, ?)''',
                       (datetime.now().isoformat(), total_trades, total_profit_loss, max_drawdown, sharpe_ratio))
        conn.commit()
    except sqlite3.Error as e:
        logging.error(f"Error while logging performance metrics: {e}")
    finally:
        conn.close()

def calculate_position_size(account_balance, risk_per_trade):
    position_size = account_balance * risk_per_trade
    return min(position_size, 10000)  # Example: MAX_POSITION_SIZE = 10000

# Function to calculate RSI using Numba
@lru_cache(maxsize=128)
def calculate_rsi(series, window=14):
    delta = np.diff(series)
    gain = np.where(delta > 0, delta, 0).sum() / window
    loss = -np.where(delta < 0, delta, 0).sum() / window
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

@jit(float64[::1](float64[::1], int32, int32, int32), nopython=True)
def calculate_macd(series, slow=26, fast=12, signal=9):
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
    global trade_count, total_loss, equity_curve

    try:
        trade_price = historical_data.iloc[-1]['price']
    except IndexError:
        logging.error("Error while fetching historical data.")
        return

    position_size = calculate_position_size(10000, 0.01)  # Example: 1% risk per trade
    trade_amount = position_size / trade_price

    if side == 'buy':
        stop_loss_price = trade_price * (1 - 0.05)  # Example: STOP_LOSS_PERCENT = 0.05
    else:
        stop_loss_price = trade_price * (1 + 0.05)

    try:
        # Simulate trade execution (use a dummy price for simplicity)
        trade_loss = 0  # Simplified example without calculating actual profit/loss
    except Exception as e:
        logging.error(f"Error while executing trade: {e}")
        return

    trade_count += 1
    total_loss += trade_loss

    if equity_curve:
        equity_curve.append(equity_curve[-1] - trade_loss)
    else:
        equity_curve = [10000 - trade_loss]  # Starting equity of 10000

    logging.info(f"Executed {side} trade for {trade_amount} at price {trade_price}, stop loss at {stop_loss_price}")

    max_drawdown = calculate_max_drawdown(np.array(equity_curve))
    sharpe_ratio = calculate_sharpe_ratio(np.array(equity_curve))
    log_performance_metrics(trade_count, total_loss, max_drawdown, sharpe_ratio)

    # Adjust backoff logic based on strategy optimization
    global backoff_duration
    backoff_duration = max(1, backoff_duration * 0.9 if side == 'buy' else backoff_duration * 1.1)
