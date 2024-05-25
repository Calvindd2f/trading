import unittest
from unittest.mock import patch, MagicMock, call
import pandas as pd
from datetime import datetime, timedelta
from trading_bot import (
    fetch_historical_data,
    process_real_time_data,
    detect_anomalies,
    execute_trade,
    on_message,
    on_open,
    on_error,
    on_close,
    SYMBOL,
    BUY_THRESHOLD,
    SELL_THRESHOLD,
    TRADE_AMOUNT,
    INITIAL_BACKOFF,
    MAX_TRADES_PER_DAY,
    MAX_LOSS_THRESHOLD,
)

class TestTradingBot(unittest.TestCase):

    def setUp(self):
        # Set up mock historical data
        self.historical_data = pd.DataFrame({
            'time': [datetime.now() - timedelta(minutes=i) for i in range(100)],
            'price': [100 + i * 0.1 for i in range(100)],
            'volume': [10 for _ in range(100)]
        })

        # Reset global variables
        daily_trades.clear()
        total_loss = 0
        backoff_duration = INITIAL_BACKOFF

    def tearDown(self):
        # Reset global variables after each test method
        daily_trades.clear()
        total_loss = 0
        backoff_duration = INITIAL_BACKOFF

    def test_anomaly_detection_pump(self):
        # Modify historical data to create a pump scenario
        self.historical_data.at[99, 'price'] = self.historical_data.at[98, 'price'] * (1 + BUY_THRESHOLD + 0.01)

        with patch.object(execute_trade, 'execute_trade') as mock_execute_trade:
            detect_anomalies()
            mock_execute_trade.assert_called_once_with("buy", TRADE_AMOUNT)

    def test_anomaly_detection_dump(self):
        # Modify historical data to create a dump scenario
        self.historical_data.at[99, 'price'] = self.historical_data.at[98, 'price'] * (1 + SELL_THRESHOLD - 0.01)

        with patch.object(execute_trade, 'execute_trade') as mock_execute_trade:
            detect_anomalies()
            mock_execute_trade.assert_called_once_with("sell", TRADE_AMOUNT)

    def test_trade_execution_safety_checks(self):
        global total_loss, daily_trades

        # Simulate maximum trades per day
        daily_trades[:] = [datetime.now() - timedelta(hours=i) for i in range(MAX_TRADES_PER_DAY)]

        with patch('requests.post') as mock_post:
            execute_trade("buy", TRADE_AMOUNT)
            self.assertFalse(mock_post.called)

        # Simulate maximum loss threshold
        total_loss = MAX_LOSS_THRESHOLD

        with patch('requests.post') as mock_post:
            execute_trade("sell", TRADE_AMOUNT)
            self.assertFalse(mock_post.called)

    def test_dynamic_backoff_adjustment(self):
        with patch('requests.post') as mock_post:
            mock_response = MagicMock()
            mock_response.elapsed.total_seconds.return_value = 0.5
            mock_response.json.return_value = {'status': 'success'}
            mock_response.raise_for_status = MagicMock()
            mock_post.return_value = mock_response

            execute_trade("buy", TRADE_AMOUNT)

            self.assertGreater(backoff_duration, INITIAL_BACKOFF)

    def test_on_message(self):
        sample_message = json.dumps({
            'time': datetime.now().isoformat() + 'Z',
            'price': '110.0',
            'volume': '5.0'
        })

        with patch('trading_bot.process_real_time_data') as mock_process:
            on_message(None, sample_message)
            mock_process.assert_called_once()

    def test_websocket_handlers(self):
        ws = MagicMock()

        # Test on_open
        on_open(ws)
        ws.send.assert_called_once_with(f"{{\"symbol\":\"{SYMBOL}\"}}")

        # Test on_error
        with patch('logging.error') as mock_log_error:
            on_error(ws, "test error")
            mock_log_error.assert_called_once_with("Error: test error")

        # Test on_close
        with patch('logging.info') as mock_log_info:
            on_close(ws, 1000, "normal closure")
            mock_log_info.assert_called_once_with("WebSocket closed")

if __name__ == '__main__':
    unittest.main()
