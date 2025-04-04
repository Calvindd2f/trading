import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Tuple
import ta
from ta.volatility import BollingerBands, AverageTrueRange
from ta.trend import MACD, EMAIndicator, IchimokuIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging

class Autoencoder(nn.Module):
    """Autoencoder for anomaly detection."""
    def __init__(self, input_dim: int, encoding_dim: int = 32):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AnomalyDetector:
    """
    A collection of state-of-the-art anomaly detection methods for market data.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = {}
        self.technical_indicators = {}
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        self.autoencoder = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate advanced technical indicators for anomaly detection.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with technical indicators
        """
        df = data.copy()
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close']).diff()
        df['price_momentum'] = df['close'].pct_change(periods=10)
        
        # Volatility features
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['volume_volatility'] = df['volume'].pct_change().rolling(window=20).std()
        atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'])
        df['atr'] = atr.average_true_range()
        
        # Trend features
        df['ema_20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
        df['ema_50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()
        df['ema_200'] = EMAIndicator(close=df['close'], window=200).ema_indicator()
        
        # Ichimoku Cloud
        ichimoku = IchimokuIndicator(high=df['high'], low=df['low'])
        df['ichimoku_a'] = ichimoku.ichimoku_a()
        df['ichimoku_b'] = ichimoku.ichimoku_b()
        df['ichimoku_base'] = ichimoku.ichimoku_base_line()
        df['ichimoku_conversion'] = ichimoku.ichimoku_conversion_line()
        
        # Momentum features
        df['rsi'] = RSIIndicator(close=df['close']).rsi()
        stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Volume features
        df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
        df['cmf'] = ChaikinMoneyFlowIndicator(high=df['high'], low=df['low'], 
                                            close=df['close'], volume=df['volume']).chaikin_money_flow()
        
        # Bollinger Bands
        bb = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
        df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # MACD
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # Market regime features
        df['trend_strength'] = abs(df['ema_20'] - df['ema_50']) / df['ema_50']
        df['volatility_regime'] = df['volatility'].rolling(window=20).mean()
        df['volume_regime'] = df['volume'].rolling(window=20).mean()
        
        return df
    
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for anomaly detection with dimensionality reduction.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            np.ndarray: Prepared features
        """
        # Calculate technical indicators
        df = self._calculate_technical_indicators(data)
        
        # Select features for anomaly detection
        features = [
            'returns', 'log_returns', 'volatility', 'volume_volatility',
            'atr', 'rsi', 'stoch_k', 'stoch_d', 'obv', 'cmf',
            'bb_width', 'bb_pct', 'macd_diff', 'trend_strength',
            'volatility_regime', 'volume_regime'
        ]
        
        # Prepare feature matrix
        X = df[features].values
        X = np.nan_to_num(X)
        
        # Apply PCA for dimensionality reduction
        if not hasattr(self.pca, 'components_'):
            X = self.pca.fit_transform(X)
        else:
            X = self.pca.transform(X)
            
        return X
    
    def train_autoencoder(self, data: pd.DataFrame, epochs: int = 50, batch_size: int = 32) -> None:
        """
        Train an autoencoder for anomaly detection.
        
        Args:
            data: Training data
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        X = self._prepare_features(data)
        X = self.scaler.fit_transform(X)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize autoencoder
        input_dim = X.shape[1]
        self.autoencoder = Autoencoder(input_dim).to(self.device)
        
        # Training
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.autoencoder.parameters())
        
        for epoch in range(epochs):
            for batch_x, _ in dataloader:
                optimizer.zero_grad()
                reconstructed = self.autoencoder(batch_x)
                loss = criterion(reconstructed, batch_x)
                loss.backward()
                optimizer.step()
        
        self.models['autoencoder'] = self.autoencoder
    
    def train_dbscan(self, data: pd.DataFrame, eps: float = 0.5, min_samples: int = 5) -> None:
        """
        Train DBSCAN for anomaly detection.
        
        Args:
            data: Training data
            eps: Maximum distance between samples
            min_samples: Minimum number of samples in a neighborhood
        """
        X = self._prepare_features(data)
        X = self.scaler.fit_transform(X)
        
        self.models['dbscan'] = DBSCAN(eps=eps, min_samples=min_samples)
        self.models['dbscan'].fit(X)
    
    def train_isolation_forest(
        self,
        data: pd.DataFrame,
        contamination: float = 0.01,
        n_estimators: int = 100,
        max_samples: int = 256
    ) -> None:
        """
        Train Isolation Forest model with adaptive parameters.
        
        Args:
            data: Training data
            contamination: Expected proportion of outliers
            n_estimators: Number of base estimators
            max_samples: Number of samples to draw
        """
        X = self._prepare_features(data)
        X = self.scaler.fit_transform(X)
        
        # Adaptive contamination based on data characteristics
        if contamination == 'auto':
            contamination = min(0.1, max(0.01, np.mean(np.abs(X)) / 10))
        
        self.models['isolation_forest'] = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=42
        )
        self.models['isolation_forest'].fit(X)
    
    def train_one_class_svm(
        self,
        data: pd.DataFrame,
        nu: float = 0.01,
        kernel: str = 'rbf',
        gamma: float = 'scale'
    ) -> None:
        """
        Train One-Class SVM model with adaptive parameters.
        
        Args:
            data: Training data
            nu: An upper bound on the fraction of training errors
            kernel: Kernel type
            gamma: Kernel coefficient
        """
        X = self._prepare_features(data)
        X = self.scaler.fit_transform(X)
        
        # Adaptive gamma based on data characteristics
        if gamma == 'auto':
            gamma = 1.0 / (X.shape[1] * X.var())
        
        self.models['one_class_svm'] = OneClassSVM(
            nu=nu,
            kernel=kernel,
            gamma=gamma
        )
        self.models['one_class_svm'].fit(X)
    
    def train_local_outlier_factor(
        self,
        data: pd.DataFrame,
        n_neighbors: int = 20,
        contamination: float = 0.01
    ) -> None:
        """
        Train Local Outlier Factor model with adaptive parameters.
        
        Args:
            data: Training data
            n_neighbors: Number of neighbors
            contamination: Expected proportion of outliers
        """
        X = self._prepare_features(data)
        X = self.scaler.fit_transform(X)
        
        # Adaptive n_neighbors based on data size
        if n_neighbors == 'auto':
            n_neighbors = min(20, max(5, int(np.sqrt(X.shape[0]))))
        
        self.models['lof'] = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True
        )
        self.models['lof'].fit(X)
    
    def detect_anomalies(
        self,
        data: pd.DataFrame,
        method: str = 'isolation_forest'
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Detect anomalies using specified method with confidence metrics.
        
        Args:
            data: Market data
            method: Anomaly detection method
            
        Returns:
            Tuple[np.ndarray, Dict[str, float]]: Anomaly scores and confidence metrics
        """
        if method not in self.models:
            raise ValueError(f"Model {method} not trained")
            
        X = self._prepare_features(data)
        X = self.scaler.transform(X)
        
        model = self.models[method]
        
        if method == 'autoencoder':
            X_tensor = torch.FloatTensor(X).to(self.device)
            with torch.no_grad():
                reconstructed = self.autoencoder(X_tensor)
                scores = torch.mean((X_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
        elif method == 'dbscan':
            scores = -model.fit_predict(X)  # -1 for outliers, 1 for inliers
        elif method == 'lof':
            scores = -model.score_samples(X)  # LOF returns negative scores
        else:
            scores = -model.score_samples(X)  # Other methods return positive scores
            
        # Calculate confidence metrics
        confidence = {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'max_score': np.max(scores),
            'min_score': np.min(scores),
            'silhouette_score': silhouette_score(X, scores > np.median(scores)) if len(np.unique(scores > np.median(scores))) > 1 else 0
        }
        
        return scores, confidence
    
    def ensemble_detect(
        self,
        data: pd.DataFrame,
        weights: Dict[str, float] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Enhanced ensemble anomaly detection using multiple methods with dynamic weighting.
        
        Args:
            data: Market data
            weights: Weights for each method
            
        Returns:
            Tuple[np.ndarray, Dict[str, float]]: Combined scores and confidence metrics
        """
        if not self.models:
            raise ValueError("No models trained")
            
        if weights is None:
            # Initialize weights based on model performance
            weights = {method: 1.0/len(self.models) for method in self.models}
            
        all_scores = []
        all_confidence = {}
        
        for method in self.models:
            scores, confidence = self.detect_anomalies(data, method)
            all_scores.append(scores * weights[method])
            all_confidence[method] = confidence
            
        # Combine scores with dynamic weighting
        combined_scores = np.sum(all_scores, axis=0)
        
        # Calculate ensemble confidence with additional metrics
        ensemble_confidence = {
            'mean_score': np.mean(combined_scores),
            'std_score': np.std(combined_scores),
            'max_score': np.max(combined_scores),
            'min_score': np.min(combined_scores),
            'method_confidence': all_confidence,
            'agreement_score': np.mean([np.corrcoef(scores, combined_scores)[0,1] for scores in all_scores]),
            'consistency_score': 1 - np.std([np.mean(scores) for scores in all_scores]) / np.mean([np.mean(scores) for scores in all_scores])
        }
        
        return combined_scores, ensemble_confidence 