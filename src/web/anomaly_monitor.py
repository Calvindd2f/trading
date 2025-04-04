import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
import plotly.graph_objects as go
from dash import html, dcc
from dash.dependencies import Input, Output
import numpy as np

from src.core.market_data_fetcher import MarketDataFetcher
from src.core.strategies.anomaly_detection import AnomalyDetectionStrategy

class AnomalyMonitor:
    """
    Real-time monitoring system for anomaly detection.
    """
    
    def __init__(
        self,
        symbol: str = "BTC/USD",
        interval: str = "1h",
        lookback: int = 100
    ):
        """
        Initialize the anomaly monitor.
        
        Args:
            symbol: Trading pair symbol
            interval: Data interval
            lookback: Number of periods to look back
        """
        self.symbol = symbol
        self.interval = interval
        self.lookback = lookback
        self.fetcher = MarketDataFetcher()
        self.strategy = AnomalyDetectionStrategy()
        self.data = None
        self.signals = None
        self.anomaly_scores = None
        
    async def update_data(self) -> None:
        """
        Update market data and anomaly detection.
        """
        # Fetch latest data
        end_time = datetime.now()
        start_time = end_time - pd.Timedelta(hours=self.lookback)
        
        self.data = self.fetcher.fetch_historical_data(
            symbol=self.symbol,
            start_time=start_time.strftime("%Y-%m-%d %H:%M:%S"),
            end_time=end_time.strftime("%Y-%m-%d %H:%M:%S"),
            interval=self.interval
        )
        
        # Generate signals
        self.signals = self.strategy.generate_signals(self.data)
        self.anomaly_scores = self.signals['anomaly_score']
        
    def create_layout(self) -> html.Div:
        """
        Create the monitoring dashboard layout.
        
        Returns:
            html.Div: Dashboard layout
        """
        return html.Div([
            html.H1(f"Anomaly Detection Monitor - {self.symbol}"),
            
            # Price and Anomaly Score Chart
            dcc.Graph(id='price-anomaly-chart'),
            
            # Technical Indicators
            html.Div([
                html.H3("Technical Indicators"),
                dcc.Graph(id='technical-indicators')
            ], style={'width': '48%', 'display': 'inline-block'}),
            
            # Anomaly Statistics
            html.Div([
                html.H3("Anomaly Statistics"),
                html.Div(id='anomaly-stats')
            ], style={'width': '48%', 'display': 'inline-block'}),
            
            # Update interval
            dcc.Interval(
                id='interval-component',
                interval=60*1000,  # Update every minute
                n_intervals=0
            )
        ])
    
    def update_charts(self) -> Dict[str, Any]:
        """
        Update the monitoring charts.
        
        Returns:
            Dict[str, Any]: Updated chart data
        """
        if self.data is None or self.signals is None:
            return {}
            
        # Create price and anomaly score chart
        fig_price = go.Figure()
        
        # Add price line
        fig_price.add_trace(go.Scatter(
            x=self.data.index,
            y=self.data['close'],
            name='Price',
            line=dict(color='blue')
        ))
        
        # Add anomaly scores
        fig_price.add_trace(go.Scatter(
            x=self.data.index,
            y=self.anomaly_scores,
            name='Anomaly Score',
            yaxis='y2',
            line=dict(color='red')
        ))
        
        # Add signals
        buy_signals = self.signals[self.signals['signal'] > 0]
        sell_signals = self.signals[self.signals['signal'] < 0]
        
        fig_price.add_trace(go.Scatter(
            x=buy_signals.index,
            y=self.data.loc[buy_signals.index, 'close'],
            mode='markers',
            name='Buy Signal',
            marker=dict(color='green', size=10)
        ))
        
        fig_price.add_trace(go.Scatter(
            x=sell_signals.index,
            y=self.data.loc[sell_signals.index, 'close'],
            mode='markers',
            name='Sell Signal',
            marker=dict(color='red', size=10)
        ))
        
        # Update layout
        fig_price.update_layout(
            title='Price and Anomaly Detection',
            yaxis=dict(title='Price'),
            yaxis2=dict(title='Anomaly Score', overlaying='y', side='right'),
            showlegend=True
        )
        
        # Create technical indicators chart
        fig_tech = go.Figure()
        
        # Add RSI
        fig_tech.add_trace(go.Scatter(
            x=self.data.index,
            y=self.signals['rsi'],
            name='RSI',
            line=dict(color='purple')
        ))
        
        # Add MACD
        fig_tech.add_trace(go.Scatter(
            x=self.data.index,
            y=self.signals['macd'],
            name='MACD',
            line=dict(color='orange')
        ))
        
        # Add Bollinger Bands
        fig_tech.add_trace(go.Scatter(
            x=self.data.index,
            y=self.signals['bb_upper'],
            name='BB Upper',
            line=dict(color='gray', dash='dash')
        ))
        
        fig_tech.add_trace(go.Scatter(
            x=self.data.index,
            y=self.signals['bb_lower'],
            name='BB Lower',
            line=dict(color='gray', dash='dash')
        ))
        
        # Update layout
        fig_tech.update_layout(
            title='Technical Indicators',
            showlegend=True
        )
        
        # Calculate anomaly statistics
        stats = {
            'mean_score': np.mean(self.anomaly_scores),
            'std_score': np.std(self.anomaly_scores),
            'max_score': np.max(self.anomaly_scores),
            'min_score': np.min(self.anomaly_scores),
            'num_anomalies': len(self.anomaly_scores[self.anomaly_scores > self.strategy.anomaly_threshold]),
            'confidence': np.mean(self.signals['confidence'])
        }
        
        return {
            'price_chart': fig_price,
            'tech_chart': fig_tech,
            'stats': stats
        }
    
    def get_anomaly_stats_html(self, stats: Dict[str, float]) -> html.Div:
        """
        Create HTML for anomaly statistics.
        
        Args:
            stats: Anomaly statistics
            
        Returns:
            html.Div: Statistics display
        """
        return html.Div([
            html.P(f"Mean Score: {stats['mean_score']:.4f}"),
            html.P(f"Std Score: {stats['std_score']:.4f}"),
            html.P(f"Max Score: {stats['max_score']:.4f}"),
            html.P(f"Min Score: {stats['min_score']:.4f}"),
            html.P(f"Number of Anomalies: {stats['num_anomalies']}"),
            html.P(f"Confidence: {stats['confidence']:.4f}")
        ])
    
    async def run_monitor(self) -> None:
        """
        Run the monitoring system.
        """
        while True:
            try:
                await self.update_data()
                await asyncio.sleep(60)  # Update every minute
            except Exception as e:
                print(f"Error in monitor: {e}")
                await asyncio.sleep(60) 