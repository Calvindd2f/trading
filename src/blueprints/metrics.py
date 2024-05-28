from flask import Blueprint, jsonify
from src.app import GlobalState

metrics_bp = Blueprint('metrics', __name__)

@metrics_bp.route('/get_metrics', methods=['GET'])
def get_metrics():
    return jsonify({
        'total_loss': GlobalState.metrics['total_loss'],
        'trade_count': GlobalState.metrics['trade_count'],
        'equity_curve': GlobalState.metrics['equity_curve']
    })
