import sqlite3
from datetime import datetime

def update_database_schema(conn):
    """
    Update the database schema by creating tables and adding a column if they don't exist.

    Parameters:
    conn (sqlite3.Connection): The database connection object.

    Returns:
    None
    """
    cursor = conn.cursor()

    # Create tables for historical data, trade logs, and performance metrics
    cursor.execute('''CREATE TABLE IF NOT EXISTS historical_data (
                        id INTEGER PRIMARY KEY,
                        time TEXT,
                        price REAL,
                        volume REAL)''')
    cursor.close()

    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS trade_logs (
                        id INTEGER PRIMARY KEY,
                        timestamp TEXT,
                        symbol TEXT,
                        side TEXT,
                        quantity REAL,
                        price REAL,
                        response TEXT)''')
    cursor.close()

    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS performance_metrics (
                        id INTEGER PRIMARY KEY,
                        timestamp TEXT,
                        total_trades INTEGER,
                        total_profit_loss REAL,
                        max_drawdown REAL,
                        sharpe_ratio REAL)''')

    # Check if the 'sharpe_ratio' column already exists before attempting to add it
    cursor.execute("SELECT name FROM pragma_table_info(performance_metrics) WHERE name='sharpe_ratio'")
    result = cursor.fetchone()
    if result is None:
        try:
            cursor.execute("ALTER TABLE performance_metrics ADD COLUMN sharpe_ratio REAL")
            conn.commit()
        except sqlite3.OperationalError as e:
            print(f"Error adding 'sharpe_ratio' column: {e}")

    cursor.close()
    conn.commit()

# Connect to the database
conn = sqlite3.connect('trading_bot.db')

# Update the database schema
update_database_schema(conn)

# Close the database connection
conn.close()
