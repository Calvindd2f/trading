simulator/:

simulator.py: Implement the simulator logic, which fake-executes trades and stores the results in a separate instance.
simulator_config.py: Store simulator-specific configuration, such as the initial capital, risk management settings, and performance metrics.
simulator_data/: Store the simulator's output data, including:
simulator_trades.csv: A record of all simulated trades, including timestamps, symbols, quantities, and prices.
simulator_performance.csv: A record of the simulator's performance metrics, such as profit/loss, drawdown, and Sharpe ratio.

How it works:

The simulator mode is enabled by setting a flag or environment variable.
The trading bot's logic is executed as usual, but instead of sending trades to an exchange, the simulator fake-executes them and stores the results in the simulator_data/ directory.
The simulator can be configured to run for a specific period, with a specific initial capital, and with various risk management settings.
The simulator's performance metrics are calculated and stored in simulator_performance.csv.

Benefits:

Test the trading bot's logic and performance without risking real capital.
Evaluate the bot's performance under different market conditions and scenarios.
Refine the bot's parameters and risk management settings before deploying it to a live trading environment.
Automatic Retraining Mode:

This mode is a great way to adapt to changing market conditions and prevent significant losses. Here's how you can implement it:

Automatic Retraining Mode Structure: