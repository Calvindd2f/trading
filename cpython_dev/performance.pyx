import numpy as np
cimport numpy as np

def calculate_max_drawdown(np.ndarray[np.float64_t, ndim=1] equity_curve):
    cdef double peak = equity_curve[0]
    cdef double max_drawdown = 0
    cdef double value, drawdown
    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return max_drawdown

def calculate_sharpe_ratio(np.ndarray[np.float64_t, ndim=1] equity_curve, double risk_free_rate=0.01):
    cdef np.ndarray[np.float64_t, ndim=1] returns = np.diff(equity_curve) / equity_curve[:-1]
    cdef np.ndarray[np.float64_t, ndim=1] excess_returns = returns - risk_free_rate / 252
    cdef double sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns)
    return sharpe_ratio * np.sqrt(252)
