from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from datetime import datetime
import json
import asyncio
from typing import Dict, List
import pandas as pd
import numpy as np

from ..core.portfolio.portfolio_manager import PortfolioManager
from ..utils.logging_config import setup_logging, TradeLogger

# Initialize FastAPI app
app = FastAPI(title="Trading Monitor")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Dash app
dash_app = dash.Dash(
    __name__,
    server=app,
    url_base_pathname="/dashboard/",
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

# Setup logging
logger = setup_logging("web_app", level="INFO")
trade_logger = TradeLogger(logger)

# Store active WebSocket connections
active_connections: List[WebSocket] = []

# Store portfolio data
portfolio_data = {
    "equity_curve": pd.Series(),
    "positions": {},
    "trades": []
}

def create_dashboard_layout():
    """Create the dashboard layout."""
    return html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Trading Dashboard", className="text-center my-4"),
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Portfolio Performance", className="card-title"),
                            dcc.Graph(id="equity-curve"),
                            dcc.Interval(id="update-interval", interval=1000)
                        ])
                    ], className="mb-4"),
                    
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Current Positions", className="card-title"),
                            html.Div(id="positions-table")
                        ])
                    ], className="mb-4"),
                    
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Recent Trades", className="card-title"),
                            html.Div(id="trades-table")
                        ])
                    ])
                ])
            ])
        ])
    ])

dash_app.layout = create_dashboard_layout()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connections."""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        while True:
            # Send portfolio updates
            await websocket.send_json({
                "type": "portfolio_update",
                "data": {
                    "equity_curve": portfolio_data["equity_curve"].to_dict(),
                    "positions": portfolio_data["positions"],
                    "trades": portfolio_data["trades"][-10:]  # Last 10 trades
                }
            })
            await asyncio.sleep(1)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        active_connections.remove(websocket)

@dash_app.callback(
    [dash.Output("equity-curve", "figure"),
     dash.Output("positions-table", "children"),
     dash.Output("trades-table", "children")],
    [dash.Input("update-interval", "n_intervals")]
)
def update_dashboard(n):
    """Update dashboard components."""
    # Update equity curve
    equity_fig = go.Figure()
    if not portfolio_data["equity_curve"].empty:
        equity_fig.add_trace(go.Scatter(
            x=portfolio_data["equity_curve"].index,
            y=portfolio_data["equity_curve"].values,
            mode="lines",
            name="Equity"
        ))
        equity_fig.update_layout(
            title="Portfolio Equity Curve",
            xaxis_title="Date",
            yaxis_title="Value"
        )
    
    # Update positions table
    positions_df = pd.DataFrame(portfolio_data["positions"])
    positions_table = dbc.Table.from_dataframe(
        positions_df,
        striped=True,
        bordered=True,
        hover=True
    ) if not positions_df.empty else html.P("No active positions")
    
    # Update trades table
    trades_df = pd.DataFrame(portfolio_data["trades"][-10:])
    trades_table = dbc.Table.from_dataframe(
        trades_df,
        striped=True,
        bordered=True,
        hover=True
    ) if not trades_df.empty else html.P("No recent trades")
    
    return equity_fig, positions_table, trades_table

def update_portfolio_data(portfolio: PortfolioManager):
    """Update portfolio data for the dashboard."""
    # Update equity curve
    portfolio_data["equity_curve"] = portfolio.equity_curve
    
    # Update positions
    portfolio_data["positions"] = {
        symbol: {
            "size": pos["size"],
            "avg_price": pos["avg_price"],
            "current_value": portfolio.calculate_position_value(symbol, pos["avg_price"])
        }
        for symbol, pos in portfolio.positions.items()
    }
    
    # Update trades
    portfolio_data["trades"] = [
        {
            "symbol": trade["symbol"],
            "action": trade["action"],
            "size": trade["size"],
            "price": trade["price"],
            "timestamp": trade["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        }
        for trade in portfolio.trades
    ]
    
    # Log updates
    trade_logger.log_performance(
        total_return=portfolio.calculate_returns()["total_return"],
        sharpe_ratio=portfolio.calculate_returns()["sharpe_ratio"],
        max_drawdown=portfolio.calculate_returns()["max_drawdown"],
        win_rate=portfolio.calculate_returns()["win_rate"]
    )

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static") 