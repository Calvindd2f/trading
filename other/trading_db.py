import sqlite3
from datetime import datetime

# Define a function to update the database schema
def update_database_schema(conn):
    cursor = conn.cursor()

    # Create tables for historical data, trade logs, and performance metrics
    cursor.execute('''CREATE TABLE IF NOT EXISTS historical_data (
                        id INTEGER PRIMARY KEY,
                        time TEXT,
                        price REAL,
                        volume REAL)''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS trade_logs (
                        id INTEGER PRIMARY KEY,
                        timestamp TEXT,
                        symbol TEXT,
                        side TEXT,
                        quantity REAL,
                        price REAL,
                        response TEXT)''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY,
                        timestamp TEXT,
                        total_trades INTEGER,
                        total_profit_loss REAL,
                        max_drawdown REAL)''')

    # Add a new column for Sharpe ratio if it doesn't exist
    cursor.execute("ALTER TABLE performance_metrics ADD COLUMN sharpe_ratio REAL")

    conn.commit()

# Connect to the database
conn = sqlite3.connect('trading_bot.db')

# Update the database schema
update_database_schema(conn)

# Close the database connection
conn.close()
