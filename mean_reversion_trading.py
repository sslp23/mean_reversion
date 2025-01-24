import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from data_retrieve import *
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from quant_functions import *

import time

def make_positions(spread, spreadMean, spreadStd, df,
                   gap = 2,
                   strat_exit = 1):
    # ticker names
    #t1, t2 = tickers[0], tickers[1]

    # calculate z score and make positions
    df["spread"] = spread
    df["zscore"] = (spread - spreadMean) / spreadStd
    df[f"positions_Pair1_Long"] = 0
    df[f"positions_Pair2_Long"] = 0
    df[f"positions_Pair1_Short"] = 0
    df[f"positions_Pair2_Short"] = 0
    
    #print(df)
    #print(time.sleep(1029))
    # Short spread
    # if spread >= +GAP (residual too positive) 
    # short t1 (-1), long t2 (1)
    # it means that t1 tends to fall while t2 tends to go up
    df.loc[df.zscore >= gap, (f"positions_Pair1_Short", f"positions_Pair2_Short")] = [
        -1,
        1,
    ]
    # Buy spread
    # if spread >= -GAP (residual too positive) 
    # long t1 (1), short t2 (-1)
    # it means that t1 tends to go up while t2 tends to fall
    df.loc[df.zscore <= -gap, (f"positions_Pair1_Long", f"positions_Pair2_Long")] = [
        1,
        -1,
    ] 
    
    #exit conditions 
    df.loc[
        df.zscore <= strat_exit, (f"positions_Pair1_Short", f"positions_Pair2_Short")
    ] = 0  # Exit short spread
    df.loc[
        df.zscore >= -strat_exit, (f"positions_Pair1_Long", f"positions_Pair2_Long")
    ] = 0  # Exit long spread

    df.ffill(inplace=True)  # ensure existing positions are carried forward unless there is an exit signal
    # build the positions
    positions_Long = df.loc[:, (f"positions_Pair1_Long", f"positions_Pair2_Long")]
    positions_Short = df.loc[:, (f"positions_Pair1_Short", f"positions_Pair2_Short")]

    # unify the positions
    positions = np.array(positions_Long) + np.array(positions_Short)
    positions = pd.DataFrame(positions)
    dailyret = df.loc[:, (f"Adj Close_Pair1", f"Adj Close_Pair2")].pct_change()
    pnl = (np.array(positions.shift()) * np.array(dailyret)).sum(axis=1)

    return positions, pnl

def process_data(data, tickers):
    # get data for each asset in the pair
    df1 = data[tickers[0]]
    df2 = data[tickers[1]]

    # make a unified df
    df = pd.merge(df1, df2, on="Date", suffixes=(f"_{tickers[0]}", f"_{tickers[1]}"))
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)

    df = df[[f"Adj Close_{tickers[0]}", f"Adj Close_{tickers[1]}"]]
    df.columns = ['Adj Close_Pair1', 'Adj Close_Pair2']
    return df

def apply_strat(data, train_idx, test_idx, gap = 2, strat_exit = 1):
    train_data = data.iloc[train_idx]
    test_data = data.iloc[test_idx]

    t1_vals = data.loc[:, f"Adj Close_Pair1"]#.iloc[trainset]
    t2_vals = data.loc[:, f"Adj Close_Pair2"]#.iloc[trainset]

    model = sm.OLS(t1_vals, t2_vals)
    results = model.fit()
    hedgeRatio = results.params.iloc[0]

    spread = t1_vals - hedgeRatio*t2_vals

    spreadMean = np.mean(spread.iloc[train_idx])

    spreadStd = np.std(spread.iloc[test_idx])

    positions, pnl = make_positions(spread, spreadMean, spreadStd, data, 
                                    gap=gap, strat_exit=strat_exit)    


    sharpe = long_only_sharpe_ratio(pnl[train_idx[1:]])
    drawdown, drawdown_duration, i = calculate_max_DD(pnl[train_idx[1:]])

    sharpe_test = long_only_sharpe_ratio(pnl[train_idx])
    drawdown_test, drawdown_duration_test, i = calculate_max_DD(pnl[test_idx])

    return positions, sharpe, sharpe_test, drawdown, drawdown_test, drawdown_duration, drawdown_duration_test

def train_model_cv(data, trainset=0.7, test_size = 0.2, gap = 2, strat_exit = 1):
    n_splits = int((1//test_size)-1)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    results_train = []
    results_test = []

    for train_index, test_index in tscv.split(data):
        #print(len(train_index), len(test_index))
        pos, sharpe_ratio, max_dd, max_ddd, sharpe_ratio_test, max_dd_test, max_ddd_test = apply_strat(data, train_index, test_index, 
                                                    gap=gap, strat_exit= strat_exit)
        #print('Train Sharpe: ', sharpe_ratio)
        #print('Train DD: ', max_dd)
        results_train.append(sharpe_ratio)

        #sharpe_ratio, max_dd, max_ddd = apply_strat(test_data, gap=gap, exit=exit)
        #print('Train Sharpe: ', sharpe_ratio)
        #print('Train DD: ', max_dd)
        
        results_test.append(sharpe_ratio_test)
        pos_train = pos.iloc[train_index]
        print('Number of trades: ', len(pos_train[pos_train[pos_train.columns[0]]!=0]))

    #pos.iloc[train_index]#.sum()
    #pos.iloc[test_index]
    #data.iloc[test_index]#.zscore.max()
    #data.iloc[train_index].zscore.max()

    print(results_test, results_train)
    print('Mean Sharpe Train: ', np.mean(results_train))
    print('Mean Sharpe Test: ', np.mean(results_test))

def main():
    tickers = ['EWN','EWZ']
    
    end = datetime.today()
    start = end - timedelta(days=5*365)
    
    data = get_data(tickers, st=start, end=end)

    data = process_data(data, tickers)

    gap = 1.5
    strat_exit = 1
    train_model_cv(data, gap=gap, strat_exit=strat_exit)
    
    #.value_counts()
    

    
    

if __name__=='__main__':
    main()