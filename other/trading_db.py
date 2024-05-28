import sqlite3
from datetime import datetime

# Define a function to update the database schema
def update_database_schema(conn):
    """
    Create the database schema if it doesn't already exist.

    :param conn: The database connection object.
    """
    cursor = conn.cursor()

    # Create tables
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS historical_data (
            id INTEGER PRIMARY KEY,
            time TEXT,
            price REAL,
            volume REAL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trade_logs (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            symbol TEXT,
            side TEXT,
            quantity REAL,
            price REAL,
            response TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            total_trades INTEGER,
            total_profit_loss REAL,
            max_drawdown REAL,
            sharpe_ratio REAL
        )
    ''')

    # Commit the changes
    conn.commit()


# Connect to the database
conn = sqlite3.connect('trading_bot.db')

# Update the database schema
update_database_schema(conn)

# Close the database connection
conn.close()
