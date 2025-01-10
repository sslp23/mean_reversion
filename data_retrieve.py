import yfinance as yf
import pandas as pd

# Define a function to fetch historical data for multiple financial symbols
# Parameters:
# - symbols: a list of stock or asset symbols to retrieve data for
# - st: the start date for the data retrieval (default is '2021-09-01')
# - end: the end date for the data retrieval (default is '2024-11-21')
# Returns:
# - A dictionary where keys are symbols and values are DataFrames containing the historical data
def get_data(symbols: list, st='2021-09-01', end='2024-11-21', how='verbose') -> dict:
    data = {}  # Initialize an empty dictionary to store the data

    # Iterate over the list of symbols
    for symbol in symbols:
        if how!='clean':
            print(symbol)  # Print the current symbol being processed
        # Fetch historical data for the current symbol using yfinance (yf) library
        # and store it in the dictionary with the symbol as the key
        
        base_df = yf.download(symbol, start=st, end=end, auto_adjust=False,  progress=False) 
        base_df = base_df.reset_index()
        #base_df['Date'] = pd.to_datetime(base_df['Date'])
        base_df.columns = base_df.columns.get_level_values(0)        
        data[symbol] = base_df
        
    return data  # Return the dictionary containing the downloaded data