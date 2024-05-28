import logging
from flask import Blueprint, jsonify
from src.data_processing import fetch_historical_data_from_db
from src.retraining.training import three_pass_training, train_model, save_model
from src.model import load_model
from src.app import GlobalState

training_bp = Blueprint('training', __name__)

@training_bp.route('/start_training', methods=['POST'])
def start_training():
    data = fetch_historical_data_from_db()
    features = ['price_change', 'volume_change', 'ma_10', 'ma_50', 'ma_200', 'ma_diff', 'std_10', 'std_50', 'momentum', 'volatility', 'rsi', 'macd']
    final_result = three_pass_training(data, features)
    logging.info(f"Final result after three passes: {final_result}")
    if final_result > 0:
        best_model = train_model(data, features)['GradientBoosting']
        save_model(best_model, 'src/optimized_pump_dump_model.pkl')
        GlobalState.model = load_model('src/optimized_pump_dump_model.pkl')
        logging.info("Retraining completed and model updated.")
        return jsonify({'status': 'success', 'message': 'Retraining completed successfully.'})
    else:
        logging.warning("Training failed to achieve positive gain/loss. Model not updated.")
        return jsonify({'status': 'failure', 'message': 'Retraining failed to achieve positive gain/loss.'})
