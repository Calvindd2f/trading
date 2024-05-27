import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

# Fetch performance metrics from the database
def fetch_performance_metrics():
    try:
        conn = sqlite3.connect('trading_bot.db')
        df = pd.read_sql_query("SELECT * FROM performance_metrics", conn)
    except sqlite3.Error as e:
        print(f"Error connecting to the database: {e}")
        return None
    else:
        conn.close()
        return df

# Generate a detailed performance report
def generate_performance_report():
    metrics_df = fetch_performance_metrics()

    if metrics_df is None:
        print("No data fetched from the database. Exiting...")
        return

    # Save summary statistics to a CSV file
    summary = metrics_df.describe()
    summary.to_csv('performance_summary.csv')

    # Generate plots and save them to files
    plt.figure(figsize=(14, 7))

    generate_plot(metrics_df['timestamp'], metrics_df['total_trades'], 'Total Trades Over Time', 'Total Trades', 'Time')
    generate_plot(metrics_df['timestamp'], metrics_df['total_profit_loss'], 'Total Profit/Loss Over Time', 'Total Profit/Loss', 'Time', color='green')
    generate_plot(metrics_df['timestamp'], metrics_df['max_drawdown'], 'Max Drawdown Over Time', 'Max Drawdown', 'Time', color='red')
    generate_plot(metrics_df['timestamp'], metrics_df['sharpe_ratio'], 'Sharpe Ratio Over Time', 'Sharpe Ratio', 'Time', color='purple')

    plt.tight_layout()
    plt.show()

def generate_plot(x, y, title, ylabel, xlabel, color=None):
    plt.subplot(2, 2, (1 + len(x.index) % 4) // 2)
    plt.plot(x, y, label='Value', color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')

# Main function to generate and save the performance report
if __name__ == "__main__":
    generate_performance_report()
