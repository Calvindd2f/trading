# Technical Indicators in Our Trading System

## Implemented Indicators

### 1. Bollinger Bands
- Used in `BollingerBandsStrategy`
- Parameters:
  - Period: 20 (default)
  - Standard Deviation: 2.0 (default)
- Signals:
  - Buy: Price crosses below lower band
  - Sell: Price crosses above upper band

### 2. Mean Reversion
- Used in `MeanReversionStrategy`
- Parameters:
  - Lookback Period: 20 (default)
  - Entry Threshold: 2.0 (default)
  - Exit Threshold: 0.5 (default)
- Signals:
  - Buy: Price significantly below mean (z-score < -entry_threshold)
  - Sell: Price significantly above mean (z-score > entry_threshold)

### 3. MACD
- Used in `MACDStrategy`
- Parameters:
  - Fast Period: 12 (default)
  - Slow Period: 26 (default)
  - Signal Period: 9 (default)
- Signals:
  - Buy: MACD crosses above signal line
  - Sell: MACD crosses below signal line

## Implementation Details

All indicators are implemented in the `src/core/strategies` directory:
- `bollinger_bands.py`
- `mean_reversion.py`
- `macd.py`

Each strategy:
- Inherits from `BaseStrategy`
- Implements `generate_signals()`
- Includes parameter optimization
- Supports position sizing
- Integrates with risk management

## Usage Example

```python
from src.core.strategies.bollinger_bands import BollingerBandsStrategy
from src.core.strategies.mean_reversion import MeanReversionStrategy
from src.core.strategies.macd import MACDStrategy

# Initialize strategies
bb_strategy = BollingerBandsStrategy(period=20, std_dev=2.0)
mr_strategy = MeanReversionStrategy(lookback_period=20, entry_threshold=2.0)
macd_strategy = MACDStrategy(fast_period=12, slow_period=26, signal_period=9)

# Generate signals
signals = strategy.generate_signals(data)
```

## Optimization

Each strategy includes parameter optimization:
- Grid search over parameter ranges
- Performance evaluation using Sharpe ratio
- Validation on out-of-sample data

## Integration

Strategies are integrated with:
- Market data fetcher
- Portfolio manager
- Risk manager
- Web dashboard