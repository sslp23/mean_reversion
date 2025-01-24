import matplotlib.pyplot as plt
from data_retrieve import *
import pandas as pd
from datetime import datetime, timedelta
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
import seaborn as sns
import numpy as np
from tqdm import tqdm

def plot_stocks(data: list, ticker1: str, ticker2: str):

    x1, y1 = data[ticker1]['Date'], data[ticker1]['Adj Close']
    x2, y2 = data[ticker2]['Date'], data[ticker2]['Adj Close']
    #print(x1)
    fig, ax1 = plt.subplots()

    # Plot the first series
    ax1.plot(x1, y1, color='blue', label=ticker1)
    ax1.set_xlabel('Date')
    
    #ax1.set_xticks(x1.values)
    #ax1.set_xticklabels(x1.values, rotation=45)
    ax1.set_ylabel(ticker1, color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.plot(x2, y2, color='red', label=ticker2)
    ax2.set_ylabel(ticker2, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    #plt.xticks(rotation=45)
    # Set a title and show the plot
    
    plt.title('Stock Historical Price')    
    fig.tight_layout()
    plt.show()

def stocks_scatter(data: list, ticker1: str, ticker2: str):
    x1, y1 = data[ticker1]['Date'], data[ticker1]['Adj Close']
    x2, y2 = data[ticker2]['Date'], data[ticker2]['Adj Close']

    # Create the regplot
    sns.regplot(x=y1, y=y2, scatter_kws={"s": 50, "alpha": 0.7}, line_kws={"color": "red"})
    
    # Customize labels and title
    plt.xlabel(ticker1)
    plt.ylabel(ticker2)
    plt.title(f'ScatterPlot with Regression Line - {ticker1} and {ticker2}')
    
    # Display the plot
    plt.tight_layout()
    plt.show()

def plot_spread(spread: list, ticker1: str, ticker2: str):
    plt.plot(spread)
    
    plt.xlabel('')
    plt.ylabel('Spread')
    plt.title(f'Spread - {ticker1} and {ticker2}')
    plt.show()

def test_cointegration(data: list, ticker1: str, ticker2: str):
    t1_vals = data[ticker1]['Adj Close'].values
    t2_vals = data[ticker2]['Adj Close'].values

    linreg = sm.OLS(t1_vals, t2_vals).fit()
    slope = linreg.params[0]
    #residuals is the spread between the assets
    residuals = t1_vals - slope * t2_vals# - linreg.intercept
    
    adf = ts.adfuller(residuals)
    return adf, residuals

def find_mult_cointegration(tickers: list, st: str, end: str):
    cointegrated_pairs = []
    for t1 in tqdm(tickers):
        for t2 in tickers:
            if t1!=t2:
                
                data = get_data([t1,t2], st=st, end=end, how='clean')
                adf, spread = test_cointegration(data, t1, t2)
                t_value = adf[0]
                c_value = adf[-2]['5%']
                p_value = adf[1]
                if t_value < c_value:
                    print(f'{t1} and {t2} are cointegrated')
                    print(f'p_value: {p_value}\nTest Value: {t_value}\nCritical Value (5%): {c_value}')
                    cointegrated_pairs.append((t1,t2))
    
    return cointegrated_pairs
    
def simple_cointegration(tickers: list, st: str, end: str):
    data = get_data(tickers, st=st, end=end)
    plot_stocks(data, tickers[1], tickers[0])
    stocks_scatter(data, tickers[1], tickers[0])
    adf, spread = test_cointegration(data, tickers[0], tickers[1])
    plot_spread(spread, tickers[0], tickers[1])
    t_value = adf[0]
    c_value = adf[-2]['5%']
    p_value = adf[1]
    if t_value < c_value:
        print(f'{tickers[0]} and {tickers[1]} are cointegrated')
        print(f'p_value: {p_value}\nTest Value: {t_value}\Critical Value (5%): {c_value}')

def main():
    tickers = ['EWH', 'EWA', 'EWK', 'EWO', 'EWC', 'EWQ', 'EWG', 'EWI', 'EWM', 
    'EWW', 'EWN', 'EWS', 'EWP', 'EWD', 'EWL', 'EWJ', 'EWY', 'EZU', 'EWU', 
    'EWZ', 'EWT', 'SPY']

    end = datetime.today() #- timedelta(days=3*365)
    start = end - timedelta(days=5*365)

    cointegrated = find_mult_cointegration(tickers, start, end)
    pd.DataFrame(cointegrated, columns=['Pair1', 'Pair2']).to_csv('cointegrated_pairs.csv', index=False)

    tickers = ['EWI', 'EWA']
    
    simple_cointegration(tickers, start, end)
    

if __name__=='__main__':
    main()