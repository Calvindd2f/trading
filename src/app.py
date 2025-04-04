import os
import logging
from flask import Flask
from src.blueprints.metrics import metrics_bp
from src.blueprints.training import training_bp
from src.model import load_model
from src.main import websocket_handler
import threading
import asyncio

app = Flask(__name__)
app.register_blueprint(metrics_bp, url_prefix="/metrics")
app.register_blueprint(training_bp, url_prefix="/training")

logging.basicConfig(level=logging.INFO)


class GlobalState:
    metrics = {"total_loss": 0, "trade_count": 0, "equity_curve": []}
    model = None


model_path = "src/optimized_pump_dump_model.pkl"

if os.path.exists(model_path):
    GlobalState.model = load_model(model_path)
else:
    logging.error(
        f"Model file not found at {model_path}. Please generate the model first."
    )


def websocket_thread():
    asyncio.run(websocket_handler())


if __name__ == "__main__":
    threading.Thread(target=websocket_thread).start()
    app.run(debug=os.getenv("DEBUG", "false").lower() == "true")
