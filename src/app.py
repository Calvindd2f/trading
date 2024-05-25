from flask import Flask, render_template, jsonify
from data_processing import fetch_historical_data_from_db
from model import load_model, predict_anomaly
import threading

app = Flask(__name__)

# Load model
model = load_model()

# Endpoint to get historical data
@app.route('/data')
def get_data():
    data = fetch_historical_data_from_db()
    return data.to_json(orient='records')

# Main route to render the homepage
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    # Start the WebSocket thread
    def start_websocket():
        import websocket
        import json

        def on_message(ws, message):
            data = json.loads(message)
            process_real_time_data(data, predict_anomaly)

        def on_error(ws, error):
            print(f"Error: {error}")

        def on_close(ws, close_status_code, close_msg):
            print("WebSocket closed")

        def on_open(ws):
            subscribe_message = json.dumps({"type": "subscribe", "channels": [{"name": "ticker", "product_ids": ["BTCUSD"]}]})
            ws.send(subscribe_message)
            print("WebSocket connection opened and subscription message sent")

        ws = websocket.WebSocketApp("wss://example.com/realtime", on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
        ws.run_forever()

    websocket_thread = threading.Thread(target=start_websocket)
    websocket_thread.start()

    # Run Flask app
    app.run(debug=True)
