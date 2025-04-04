import logging
from flask import Blueprint, jsonify
from src.data_processing import fetch_historical_data_from_db
from src.retraining.training import three_pass_training, train_model, save_model
from src.model import load_model
from src.app import GlobalState
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json
from datetime import datetime, timedelta
from ..core.market_data.market_data_fetcher import MarketDataFetcher
from ..core.strategies.bollinger_bands import BollingerBandsStrategy
from ..core.strategies.mean_reversion import MeanReversionStrategy
from ..core.strategies.macd import MACDStrategy
from ..core.portfolio.portfolio_manager import PortfolioManager
from ..core.risk.risk_manager import RiskManager
from ..utils.logging_config import setup_logging, TradeLogger

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

def run_training_session(
    symbols: List[str],
    start_date: str,
    end_date: str,
    initial_capital: float = 100000.0,
    risk_per_trade: float = 0.02,
    output_dir: Optional[Path] = None
) -> Dict:
    """
    Run a training session to test and optimize multiple trading strategies.
    
    Args:
        symbols: List of trading symbols
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        initial_capital: Initial capital for backtesting
        risk_per_trade: Risk per trade as a fraction of capital
        output_dir: Directory to save results
        
    Returns:
        Dictionary containing training results
    """
    # Setup logging
    logger = setup_logging("training", level="INFO")
    trade_logger = TradeLogger(logger)
    
    # Create output directory if specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize market data fetcher
    data_fetcher = MarketDataFetcher()
    
    # Initialize strategies
    strategies = {
        "Bollinger Bands": BollingerBandsStrategy(params={
            "account_value": initial_capital,
            "risk_per_trade": risk_per_trade
        }),
        "Mean Reversion": MeanReversionStrategy(params={
            "account_value": initial_capital,
            "risk_per_trade": risk_per_trade
        }),
        "MACD": MACDStrategy(params={
            "account_value": initial_capital,
            "risk_per_trade": risk_per_trade
        })
    }
    
    # Initialize portfolio and risk managers
    portfolio = PortfolioManager(initial_capital=initial_capital)
    risk_manager = RiskManager(
        max_position_size=0.1,  # 10% of portfolio per position
        max_drawdown=0.2,  # 20% max drawdown
        max_leverage=2.0  # 2x leverage
    )
    
    # Results storage
    results = {
        "symbols": symbols,
        "start_date": start_date,
        "end_date": end_date,
        "initial_capital": initial_capital,
        "risk_per_trade": risk_per_trade,
        "strategies": {}
    }
    
    # Run training for each symbol
    for symbol in symbols:
        logger.info(f"Processing {symbol}")
        results["strategies"][symbol] = {}
        
        # Fetch historical data
        data = data_fetcher.get_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            interval="1d"
        )
        
        if data.empty:
            logger.warning(f"No data found for {symbol}")
            continue
        
        # Split data into training and validation sets
        split_date = datetime.strptime(end_date, "%Y-%m-%d") - timedelta(days=365)
        train_data = data[data.index <= split_date]
        val_data = data[data.index > split_date]
        
        # Test each strategy
        for strategy_name, strategy in strategies.items():
            logger.info(f"Testing {strategy_name} strategy")
            
            # Optimize parameters on training data
            logger.info("Optimizing parameters...")
            optimal_params = strategy.optimize_parameters(train_data)
            
            # Apply optimal parameters
            for param, value in optimal_params.items():
                if hasattr(strategy, param):
                    setattr(strategy, param, value)
            
            # Generate signals on validation data
            signals = strategy.generate_signals(val_data)
            
            # Simulate trading
            portfolio.reset()
            trades = []
            
            for i in range(1, len(signals)):
                current_signal = signals.iloc[i]['signal']
                prev_signal = signals.iloc[i-1]['signal']
                
                if current_signal != prev_signal:
                    price = signals.iloc[i]['Close']
                    size = signals.iloc[i]['position_size']
                    
                    # Check risk limits
                    if not risk_manager.check_position_risk(
                        symbol=symbol,
                        size=size,
                        price=price,
                        portfolio=portfolio
                    ):
                        continue
                    
                    # Execute trade
                    if current_signal == 1:  # Buy
                        portfolio.enter_position(
                            symbol=symbol,
                            size=size,
                            price=price,
                            timestamp=signals.index[i]
                        )
                    elif current_signal == -1:  # Sell
                        portfolio.exit_position(
                            symbol=symbol,
                            size=size,
                            price=price,
                            timestamp=signals.index[i]
                        )
                    
                    trades.append({
                        "timestamp": signals.index[i],
                        "symbol": symbol,
                        "action": "buy" if current_signal == 1 else "sell",
                        "size": size,
                        "price": price
                    })
            
            # Calculate performance metrics
            performance = portfolio.calculate_returns()
            
            # Store results
            results["strategies"][symbol][strategy_name] = {
                "parameters": optimal_params,
                "trades": trades,
                "performance": performance
            }
            
            # Log performance
            trade_logger.log_performance(
                total_return=performance["total_return"],
                sharpe_ratio=performance["sharpe_ratio"],
                max_drawdown=performance["max_drawdown"],
                win_rate=performance["win_rate"]
            )
    
    # Save results if output directory specified
    if output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"training_results_{timestamp}.json"
        
        # Convert datetime objects to strings for JSON serialization
        def datetime_handler(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj
        
        with open(results_file, "w") as f:
            json.dump(results, f, default=datetime_handler, indent=4)
        
        logger.info(f"Results saved to {results_file}")
    
    return results

def check_gpu_availability() -> bool:
    """
    Check if NVIDIA GPU is available for training.
    
    Returns:
        bool: True if NVIDIA GPU is available
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def train_models_from_results(
    results: Dict,
    model_type: str = "ensemble",
    output_dir: Optional[Path] = None,
    use_gpu: bool = True,
    batch_size: int = 1024,
    num_workers: int = 4
) -> Dict:
    """
    Train models using the results from strategy backtesting with GPU acceleration.
    
    Args:
        results: Dictionary containing training results
        model_type: Type of model to train ("ensemble", "xgboost", "random_forest", "neural_net")
        output_dir: Directory to save trained models
        use_gpu: Whether to use GPU acceleration if available
        batch_size: Batch size for neural network training
        num_workers: Number of workers for data loading
        
    Returns:
        Dictionary containing trained models and their performance
    """
    logger = setup_logging("model_training", level="INFO")
    
    # Check GPU availability
    gpu_available = check_gpu_availability() if use_gpu else False
    if gpu_available:
        logger.info("NVIDIA GPU detected. Using GPU acceleration for training.")
    else:
        logger.info("No GPU detected or GPU usage disabled. Using CPU for training.")
    
    # Create output directory if specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare training data with parallel processing
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    
    def process_trade_data(args):
        symbol, strategy_name, strategy_data = args
        trades = strategy_data["trades"]
        performance = strategy_data["performance"]
        
        features = []
        for trade in trades:
            features.append({
                "symbol": symbol,
                "strategy": strategy_name,
                "price": trade["price"],
                "size": trade["size"],
                "action": 1 if trade["action"] == "buy" else -1,
                "timestamp": trade["timestamp"],
                **strategy_data["parameters"]
            })
        
        return features, [1 if performance["total_return"] > 0 else 0] * len(trades)
    
    # Process data in parallel
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        args = [(symbol, strategy_name, strategy_data) 
                for symbol, strategies in results["strategies"].items()
                for strategy_name, strategy_data in strategies.items()]
        
        results = list(executor.map(process_trade_data, args))
        features = [f for r in results for f in r[0]]
        y = [label for r in results for label in r[1]]
    
    # Convert to DataFrame
    X = pd.DataFrame(features)
    
    # Add time-based features
    X["hour"] = pd.to_datetime(X["timestamp"]).dt.hour
    X["day_of_week"] = pd.to_datetime(X["timestamp"]).dt.dayofweek
    X["month"] = pd.to_datetime(X["timestamp"]).dt.month
    
    # One-hot encode categorical features
    X = pd.get_dummies(X, columns=["symbol", "strategy", "action"])
    
    # Split into training and validation sets
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Train models based on type
    models = {}
    
    if model_type == "ensemble":
        from sklearn.ensemble import VotingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        
        # Initialize base models with GPU support
        lr = LogisticRegression(max_iter=1000, n_jobs=-1)
        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        
        # Configure XGBoost for GPU if available
        xgb_params = {
            'n_estimators': 100,
            'tree_method': 'gpu_hist' if gpu_available else 'hist',
            'gpu_id': 0 if gpu_available else None,
            'predictor': 'gpu_predictor' if gpu_available else 'cpu_predictor'
        }
        xgb = XGBClassifier(**xgb_params)
        
        # Create ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('lr', lr),
                ('rf', rf),
                ('xgb', xgb)
            ],
            voting='soft',
            n_jobs=-1
        )
        
        # Train ensemble
        ensemble.fit(X_train, y_train)
        models["ensemble"] = ensemble
        
    elif model_type == "xgboost":
        from xgboost import XGBClassifier
        
        # Configure XGBoost for GPU
        xgb_params = {
            'n_estimators': 100,
            'tree_method': 'gpu_hist' if gpu_available else 'hist',
            'gpu_id': 0 if gpu_available else None,
            'predictor': 'gpu_predictor' if gpu_available else 'cpu_predictor'
        }
        xgb = XGBClassifier(**xgb_params)
        xgb.fit(X_train, y_train)
        models["xgboost"] = xgb
        
    elif model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
        rf.fit(X_train, y_train)
        models["random_forest"] = rf
        
    elif model_type == "neural_net":
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        
        # Convert data to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train.values)
        y_train_tensor = torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val.values)
        y_val_tensor = torch.FloatTensor(y_val)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        # Define neural network
        class TradingNet(nn.Module):
            def __init__(self, input_size):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(input_size, 256),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                return self.network(x)
        
        # Initialize model and move to GPU if available
        model = TradingNet(X_train.shape[1])
        if gpu_available:
            model = model.cuda()
        
        # Training setup
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        num_epochs = 50
        for epoch in range(num_epochs):
            model.train()
            for batch_X, batch_y in train_loader:
                if gpu_available:
                    batch_X, batch_y = batch_X.cuda(), batch_y.cuda()
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y.unsqueeze(1))
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for batch_X, batch_y in val_loader:
                    if gpu_available:
                        batch_X, batch_y = batch_X.cuda(), batch_y.cuda()
                    outputs = model(batch_X)
                    val_loss += criterion(outputs, batch_y.unsqueeze(1)).item()
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss/len(val_loader):.4f}")
        
        models["neural_net"] = model
    
    # Evaluate models
    model_performance = {}
    for model_name, model in models.items():
        if model_name == "neural_net":
            # Neural network evaluation
            model.eval()
            with torch.no_grad():
                if gpu_available:
                    X_val_tensor = X_val_tensor.cuda()
                y_pred_proba = model(X_val_tensor).cpu().numpy()
                y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            # Traditional model evaluation
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred),
            "recall": recall_score(y_val, y_pred),
            "f1": f1_score(y_val, y_pred),
            "roc_auc": roc_auc_score(y_val, y_pred_proba)
        }
        
        model_performance[model_name] = metrics
        
        # Log performance
        logger.info(f"{model_name} performance:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
    
    # Save models if output directory specified
    if output_dir:
        import joblib
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, model in models.items():
            if model_name == "neural_net":
                # Save PyTorch model
                model_file = output_dir / f"{model_name}_{timestamp}.pt"
                torch.save(model.state_dict(), model_file)
            else:
                # Save scikit-learn model
                model_file = output_dir / f"{model_name}_{timestamp}.joblib"
                joblib.dump(model, model_file)
            logger.info(f"Saved {model_name} to {model_file}")
        
        # Save performance metrics
        metrics_file = output_dir / f"model_performance_{timestamp}.json"
        with open(metrics_file, "w") as f:
            json.dump(model_performance, f, indent=4)
        logger.info(f"Saved performance metrics to {metrics_file}")
    
    return {
        "models": models,
        "performance": model_performance,
        "feature_importance": {
            model_name: dict(zip(X.columns, model.feature_importances_))
            for model_name, model in models.items()
            if hasattr(model, "feature_importances_")
        }
    }

if __name__ == "__main__":
    # Example usage
    symbols = ["AAPL", "MSFT", "GOOGL"]
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    initial_capital = 100000.0
    risk_per_trade = 0.02
    output_dir = Path("results/training")
    
    # Run training session
    results = run_training_session(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital,
        risk_per_trade=risk_per_trade,
        output_dir=output_dir
    )
    
    # Train models on results with GPU acceleration
    model_results = train_models_from_results(
        results=results,
        model_type="ensemble",  # or "neural_net" for deep learning
        output_dir=output_dir / "models",
        use_gpu=True,
        batch_size=1024,
        num_workers=4
    )
