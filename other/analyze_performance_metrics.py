import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# Fetch performance metrics from the database
def fetch_performance_metrics() -> pd.DataFrame:
    """
    Fetches the performance metrics from the trading_bot.db SQLite database.

    Returns:
        A pandas DataFrame containing the performance metrics.
    """
    try:
        conn = sqlite3.connect('trading_bot.db')
        df = pd.read_sql_query("SELECT * FROM performance_metrics", conn)
        conn.close()
        return df
    except sqlite3.Error as e:
        print(f"Error fetching performance metrics: {e}")
        return pd.DataFrame()

# Analyze performance metrics
def analyze_performance_metrics(metrics_df: pd.DataFrame):
    """
    Analyzes the given performance metrics DataFrame.

    Args:
        metrics_df (pandas.DataFrame): The DataFrame containing the performance metrics.
    """
    # Summary statistics
    print("Summary Statistics:\n")
    print(metrics_df.describe())

    # Plot performance metrics over time
    plt.figure(figsize=(14, 7))

    plt.subplot(2, 2, 1)
    plt.plot(metrics_df['timestamp'], metrics_df['total_trades'], label='Total Trades')
    plt.title('Total Trades Over Time')
    plt.xlabel('Time')
    plt.ylabel('Total Trades')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(metrics_df['timestamp'], metrics_df['total_profit_loss'], label='Total Profit/Loss', color='green')
    plt.title('Total Profit/Loss Over Time')
    plt.xlabel('Time')
    plt.ylabel('Total Profit/Loss')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(metrics_df['timestamp'], metrics_df['max_drawdown'], label='Max Drawdown', color='red')
    plt.title('Max Drawdown Over Time')
    plt.xlabel('Time')
    plt.ylabel('Max Drawdown')
    plt.legend()

    plt.subplot(2, 2, 4)
    plt.plot(metrics_df['timestamp'], metrics_df['sharpe_ratio'], label='Sharpe Ratio', color='purple')
    plt.title('Sharpe Ratio Over Time')
    plt.xlabel('Time')
    plt.ylabel('Sharpe Ratio')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Calculate and print cumulative profit
def print_cumulative_profit(metrics_df: pd.DataFrame):
    """
    Calculates and prints the cumulative profit from the performance metrics DataFrame.

    Args:
        metrics_df (pandas.DataFrame): The DataFrame containing the performance metrics.
    """
    cumulative_profit_df = metrics_df.copy()
    cumulative_profit_df['cumulative_profit_loss'] = cumulative_profit_df['total_profit_loss'].cumsum()
    print("\nCumulative Profit:")
    print(cumulative_profit_df[['timestamp', 'cumulative_profit_loss']])

# Main function to analyze and report performance metrics
if __name__ == "__main__":
    metrics_df = fetch_performance_metrics()
    if not metrics_df.empty:
        analyze_performance_metrics(metrics_df)
        print_cumulative_profit(metrics_df)
