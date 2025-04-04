# Market Data Integration

## Current Implementation

### MarketDataFetcher
- Location: `src/core/market_data_fetcher.py`
- Features:
  - Real-time data streaming
  - Historical data fetching
  - Data preprocessing
  - Error handling and retries

### Supported Data Sources
1. CoinGecko API
   - Market data
   - Price history
   - Volume data
   - Market cap

2. Custom Data Sources
   - WebSocket connections
   - REST APIs
   - Local data storage

### Data Processing
```python
from src.core.market_data_fetcher import MarketDataFetcher

# Initialize fetcher
fetcher = MarketDataFetcher()

# Fetch historical data
historical_data = fetcher.fetch_historical_data(
    symbol="BTC/USD",
    start_time="2024-01-01",
    end_time="2024-03-31",
    interval="1h"
)

# Stream real-time data
async def handle_realtime_data(data):
    # Process incoming market data
    pass

fetcher.stream_realtime_data(
    symbol="BTC/USD",
    callback=handle_realtime_data
)
```

### Data Storage
- Local CSV files
- In-memory caching
- Database integration (optional)

### Error Handling
- Automatic retries
- Rate limiting
- Connection recovery
- Data validation

### Integration
- Strategy execution
- Portfolio management
- Risk assessment
- Performance monitoring

## Configuration
```yaml
market_data:
  api_key: "your_api_key"
  rate_limit: 100  # requests per minute
  retry_count: 3
  retry_delay: 5  # seconds
  cache_duration: 3600  # seconds
```

## Future Enhancements
1. Additional data sources
2. Advanced caching mechanisms
3. Data quality monitoring
4. Performance optimization
