import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

# Fetch performance metrics from the database
def fetch_performance_metrics():
    conn = sqlite3.connect('trading_bot.db')
    df = pd.read_sql_query("SELECT * FROM performance_metrics", conn)
    conn.close()
    return df

# Generate a detailed performance report
def generate_performance_report():
    metrics_df = fetch_performance_metrics()
    
    # Save summary statistics to a CSV file
    summary = metrics_df.describe()
    summary.to_csv('performance_summary.csv')
    
    # Generate plots and save them to files
    plt.figure(figsize=(14, 7))
    
    plt.subplot(2, 2, 1)
    plt.plot(metrics_df['timestamp'], metrics_df['total_trades'], label='Total Trades')
    plt.title('Total Trades Over Time')
    plt.xlabel('Time')
    plt.ylabel('Total Trades')
    plt.legend()
    plt.savefig('total_trades_over_time.png')
    
    plt.subplot(2, 2, 2)
    plt.plot(metrics_df['timestamp'], metrics_df['total_profit_loss'], label='Total Profit/Loss', color='green')
    plt.title('Total Profit/Loss Over Time')
    plt.xlabel('Time')
    plt.ylabel('Total Profit/Loss')
    plt.legend()
    plt.savefig('total_profit_loss_over_time.png')
    
    plt.subplot(2, 2, 3)
    plt.plot(metrics_df['timestamp'], metrics_df['max_drawdown'], label='Max Drawdown', color='red')
    plt.title('Max Drawdown Over Time')
    plt.xlabel('Time')
    plt.ylabel('Max Drawdown')
    plt.legend()
    plt.savefig('max_drawdown_over_time.png')
    
    plt.subplot(2, 2, 4)
    plt.plot(metrics_df['timestamp'], metrics_df['sharpe_ratio'], label='Sharpe Ratio', color='purple')
    plt.title('Sharpe Ratio Over Time')
    plt.xlabel('Time')
    plt.ylabel('Sharpe Ratio')
    plt.legend()
    plt.savefig('sharpe_ratio_over_time.png')
    
    plt.tight_layout()
    plt.show()

# Main function to generate and save the performance report
if __name__ == "__main__":
    generate_performance_report()
