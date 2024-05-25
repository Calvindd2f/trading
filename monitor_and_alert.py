from twilio.rest import Client

# Twilio configuration (replace with your own Twilio credentials)
account_sid = 'your_account_sid'
auth_token = 'your_auth_token'
twilio_phone_number = 'your_twilio_phone_number'
target_phone_number = 'your_phone_number'

# Initialize Twilio client
client = Client(account_sid, auth_token)

# Send SMS alert
def send_alert(message):
    client.messages.create(
        body=message,
        from_=twilio_phone_number,
        to=target_phone_number
    )

# Monitor performance metrics and send alerts
def monitor_and_alert():
    metrics_df = fetch_performance_metrics()
    
    # Check for significant events (e.g., max drawdown exceeds threshold)
    latest_metrics = metrics_df.iloc[-1]
    if latest_metrics['max_drawdown'] > 0.2:  # Example threshold for max drawdown
        send_alert(f"Alert: Max Drawdown Exceeded! Current Max Drawdown: {latest_metrics['max_drawdown']:.2%}")
    
    # Example: Alert for negative Sharpe Ratio
    if latest_metrics['sharpe_ratio'] < 0:
        send_alert(f"Alert: Negative Sharpe Ratio! Current Sharpe Ratio: {latest_metrics['sharpe_ratio']:.2f}")

# Main function to continuously monitor and alert
if __name__ == "__main__":
    monitor_and_alert()