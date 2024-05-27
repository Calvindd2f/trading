import time
import os
import pandas as pd
from twilio.rest import Client

# Twilio configuration (replace with your own Twilio credentials)
account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
twilio_phone_number = os.environ.get("TWILIO_PHONE_NUMBER")
target_phone_number = os.environ.get("PHONE_NUMBER")

# Initialize Twilio client
client = Client(account_sid, auth_token)

# Send SMS alert
def send_alert(message):
    try:
        client.messages.create(
            body=message,
            from_=twilio_phone_number,
            to=target_phone_number
        )
    except Exception as e:
        print(f"Error sending alert: {e}")

# Fetch performance metrics (implement this according to your data source)
def fetch_performance_metrics():
    # Placeholder implementation, replace with actual data fetching
    try:
        return pd.DataFrame({
            'max_drawdown': [0.15],
            'sharpe_ratio': [0.8]
        })
    except Exception as e:
        print(f"Error fetching performance metrics: {e}")
        return pd.DataFrame()

# Monitor performance metrics and send alerts
def monitor_and_alert():
    while True:
        metrics_df = fetch_performance_metrics()

        if not metrics_df.empty:
            # Check for significant events (e.g., max drawdown exceeds threshold)
            latest_metrics = metrics_df.iloc[-1]
            if latest_metrics['max_drawdown'] > 0.2:  # Example threshold for max drawdown
                send_alert(f"Alert: Max Drawdown Exceeded! Current Max Drawdown: {latest_metrics['max_drawdown']:.2%}. Details: {metrics_df}")

            # Example: Alert for negative Sharpe Ratio
            if latest_metrics['sharpe_ratio'] < 0:
                send_alert(f"Alert: Negative Sharpe Ratio! Current Sharpe Ratio: {latest_metrics['sharpe_ratio']:.2f}. Details: {metrics_df}")

        # Wait for some time before the next check
        time.sleep(60 * 5)  # Check every 5 minutes

# Main function to continuously monitor and alert
if __name__ == "__main__":
    monitor_and_alert()
