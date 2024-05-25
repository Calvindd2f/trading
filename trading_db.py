import sqlite3
from datetime import datetime

# Create a connection to the SQLite database
conn = sqlite3.connect('trading_bot.db')
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

conn.commit()
conn.close()
