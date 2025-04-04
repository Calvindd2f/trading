import logging
import logging.handlers
from pathlib import Path
from typing import Optional
import sys
from config.config import LOGGING_CONFIG

def setup_logging(
    name: str,
    log_file: Optional[Path] = None,
    level: str = "INFO",
    format_str: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        format_str: Log format string
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatters
    if format_str is None:
        format_str = LOGGING_CONFIG["format"]
    formatter = logging.Formatter(format_str)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if log file is specified
    if log_file:
        # Create log directory if it doesn't exist
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

class TradeLogger:
    def __init__(self, logger: logging.Logger):
        """
        Initialize trade logger.
        
        Args:
            logger: Base logger instance
        """
        self.logger = logger
        
    def log_trade(
        self,
        symbol: str,
        action: str,
        size: float,
        price: float,
        timestamp: str,
        pnl: Optional[float] = None
    ) -> None:
        """
        Log trade information.
        
        Args:
            symbol: Trading symbol
            action: Trade action
            size: Trade size
            price: Trade price
            timestamp: Trade timestamp
            pnl: Profit/loss if available
        """
        msg = f"Trade: {action.upper()} {size} {symbol} @ {price}"
        if pnl is not None:
            msg += f" (PnL: {pnl:.2f})"
        self.logger.info(msg)
        
    def log_position(
        self,
        symbol: str,
        size: float,
        avg_price: float,
        current_price: float
    ) -> None:
        """
        Log position information.
        
        Args:
            symbol: Trading symbol
            size: Position size
            avg_price: Average entry price
            current_price: Current price
        """
        pnl = (current_price - avg_price) * size
        pnl_pct = (current_price - avg_price) / avg_price * 100
        
        self.logger.info(
            f"Position: {symbol} - Size: {size}, Avg Price: {avg_price:.2f}, "
            f"Current: {current_price:.2f}, PnL: {pnl:.2f} ({pnl_pct:.2f}%)"
        )
        
    def log_performance(
        self,
        total_return: float,
        sharpe_ratio: float,
        max_drawdown: float,
        win_rate: float
    ) -> None:
        """
        Log performance metrics.
        
        Args:
            total_return: Total return
            sharpe_ratio: Sharpe ratio
            max_drawdown: Maximum drawdown
            win_rate: Win rate
        """
        self.logger.info(
            f"Performance - Return: {total_return:.2%}, Sharpe: {sharpe_ratio:.2f}, "
            f"Max DD: {max_drawdown:.2%}, Win Rate: {win_rate:.2%}"
        )
        
    def log_error(self, error: Exception, context: Optional[str] = None) -> None:
        """
        Log error information.
        
        Args:
            error: Exception instance
            context: Additional context
        """
        msg = f"Error: {str(error)}"
        if context:
            msg += f" (Context: {context})"
        self.logger.error(msg, exc_info=True) 