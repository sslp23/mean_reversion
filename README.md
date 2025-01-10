# Mean Reversion Trading Strategies

This repository is dedicated to building and implementing algorithmic trading strategies based on the **mean reversion** concept. Mean reversion assumes that asset prices and returns eventually move back toward their historical average or mean.

## Features

### Current Files

#### `data_retrieve.py`
This script retrieves financial data from Yahoo Finance. It can be used to collect historical data for stocks, indices, ETFs, and currencies.

#### `cointegration.py`
This script identifies cointegrated pairs of assets, which is essential for building mean reversion strategies. It contains the following key functions:

- **`find_mult_cointegration`**: Tests cointegration between multiple tickers to identify potential pairs for trading.
- **`simple_cointegration`**: Analyzes the cointegration between a single pair of assets.

### Upcoming Features

A new file is under development to build the mean reversion strategy and generate trading signals. This will include:
- Defining entry and exit points for trades.
- Generating trading signals based on z-scores or other statistical methods.
- Backtesting strategies to evaluate performance.

## Usage

1. Retrieve data using `data_retrieve.py`:
   ```python
   from data_retrieve import get_data
   data = get_data("AAPL", start_date="2020-01-01", end_date="2023-01-01")
   ```

2. Find cointegrated pairs using `cointegration.py`:
   ```python
   from cointegration import find_mult_cointegration, simple_cointegration

   # Test cointegration for multiple tickers
   tickers = ['EWA', 'EWK', 'EWO', 'EWC', 'EWQ', 'EWG', 'EWH', 'EWI', 'EWJ', 'EWM', 
       'EWW', 'EWN', 'EWS', 'EWP', 'EWD', 'EWL', 'EWJ', 'EWY', 'EZU', 'EWU', 
       'EWZ', 'EWT', 'SPY']
   cointegrated = find_mult_cointegration(tickers, start, end)

   # Analyze a single pair
   tickers = ['EZU', 'EWD']
   simple_cointegration(tickers, start, end)
   ```
