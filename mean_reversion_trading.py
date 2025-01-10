import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from data_retrieve import *
from datetime import datetime, timedelta
from quant_functions import *

def make_positions(spread, spreadMean, spreadStd, df, tickers,
                   gap = 2,
                   exit = 1):
    # ticker names
    t1, t2 = tickers[0], tickers[1]

    # calculate z score and make positions
    df["zscore"] = (spread - spreadMean) / spreadStd
    df[f"positions_{t1}_Long"] = 0
    df[f"positions_{t2}_Long"] = 0
    df[f"positions_{t1}_Short"] = 0
    df[f"positions_{t2}_Short"] = 0

    # Short spread
    # if spread >= +GAP (residual too positive) 
    # short t1 (-1), long t2 (1)
    # it means that t1 tends to fall while t2 tends to go up
    df.loc[df.zscore >= gap, (f"positions_{t1}_Short", f"positions_{t2}_Short")] = [
        -1,
        1,
    ]

    # Buy spread
    # if spread >= -GAP (residual too positive) 
    # long t1 (1), short t2 (-1)
    # it means that t1 tends to go up while t2 tends to fall
    df.loc[df.zscore <= -gap, (f"positions_{t1}_Long", f"positions_{t2}_Long")] = [
        1,
        -1,
    ] 
    
    #exit conditions 
    df.loc[
        df.zscore <= exit, (f"positions_{t1}_Short", f"positions_{t2}_Short")
    ] = 0  # Exit short spread
    df.loc[
        df.zscore >= -exit, (f"positions_{t1}_Long", f"positions_{t2}_Long")
    ] = 0  # Exit long spread

    df.ffill(inplace=True)  # ensure existing positions are carried forward unless there is an exit signal
    # build the positions
    positions_Long = df.loc[:, (f"positions_{t1}_Long", f"positions_{t2}_Long")]
    positions_Short = df.loc[:, (f"positions_{t1}_Short", f"positions_{t2}_Short")]

    # unify the positions
    positions = np.array(positions_Long) + np.array(positions_Short)
    positions = pd.DataFrame(positions)
    dailyret = df.loc[:, (f"Adj Close_{t1}", f"Adj Close_{t2}")].pct_change()
    pnl = (np.array(positions.shift()) * np.array(dailyret)).sum(axis=1)

    return positions, pnl

def train_model(data, tickers, trainset=0.7, gap=2, exit=1):
    # get data for each asset in the pair
    df1 = data[tickers[0]]
    df2 = data[tickers[1]]

    # make a unified df
    df = pd.merge(df1, df2, on="Date", suffixes=(f"_{tickers[0]}", f"_{tickers[1]}"))
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)

    # calculate the train and test sets
    length_train = np.ceil(len(df1)*trainset)
    trainset = np.arange(0, length_train)
    testset = np.arange(trainset.shape[0], df.shape[0])
    
    # get the adjusted close value for each asset
    t1_vals = df.loc[:, f"Adj Close_{tickers[0]}"].iloc[trainset]
    t2_vals = df.loc[:, f"Adj Close_{tickers[1]}"].iloc[trainset]
    
    # make the linear regression to find the slope (hedge ratio)
    # lin reg -> (y, x)
    # t1_vals = b*t2_vals + c 
    
    model = sm.OLS(t1_vals, t2_vals)
    results = model.fit()
    hedgeRatio = results.params[0]

    t1_vals = df.loc[:, f"Adj Close_{tickers[0]}"]
    t2_vals = df.loc[:, f"Adj Close_{tickers[1]}"]
    
    # calculate the spread (residuals)
    # residuals (spread) = t1_vals (actual value) - (b*t2_vals) (predicted value)
    spread = t1_vals - hedgeRatio*t2_vals

    plt.plot(spread)
    plt.plot(spread.iloc[trainset], label = 'Train', color='r')
    plt.plot(spread.iloc[testset], label = 'Test', color='b')
    plt.legend()
    plt.title(f'Spread - {tickers[0]} and {tickers[1]}')
    plt.show()
    plt.close()
    
    # mean spread (used to calculate z - score) 
    # need to use only trainset values (parameter)
    spreadMean = np.mean(spread.iloc[trainset])

    spreadStd = np.std(spread.iloc[trainset])

    positions, pnl = make_positions(spread, spreadMean, spreadStd, df, tickers, gap=gap, exit=exit)
    #return positions, dailyret, pnl

    trainset = np.array(trainset, dtype=int)
    testset = np.array(testset, dtype=int)

    sharpeTrainset = long_only_sharpe_ratio(pnl[trainset[1:]])
    print("Sharpe Trainset: ", sharpeTrainset)

    sharpeTestset = long_only_sharpe_ratio(pnl[testset])
    print("Sharpe Testset: ", sharpeTestset)

    drawdown, drawdown_duration, i = calculate_max_DD(pnl[trainset[1:]])
    print(f"Max Drawdown (Train): {drawdown}\nMax Drawdown Duration (Train): {drawdown_duration}")
    
    plt.plot(np.cumsum(pnl[1:]))
    plt.title(f'Cumulative Returns - mean reversion {tickers[0]} and {tickers[1]}')
    plt.show()
    plt.close()

#def backtest(data, tickers)

def main():
    tickers = ['EWA', 'EWU']
    
    end = datetime.today()
    start = end - timedelta(days=5*365)
    
    data = get_data(tickers, st=start, end=end)

    train_model(data, tickers, gap=1.75, exit=1)


    
    

if __name__=='__main__':
    main()