import numpy as np
from performance import calculate_max_drawdown, calculate_sharpe_ratio

# Example usage
equity_curve = np.array([10000, 10100, 10200, 9900, 9950])
max_drawdown = calculate_max_drawdown(equity_curve)
sharpe_ratio = calculate_sharpe_ratio(equity_curve)
print(f"Max Drawdown: {max_drawdown}")
print(f"Sharpe Ratio: {sharpe_ratio}")
