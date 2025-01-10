import numpy as np

def long_only_sharpe_ratio(rets):    
    #EXCESS DAILY -> STRAT RETURNS - FINANCING COSTS (RISK FREE)
    excess_ret = rets - 0.04/252

    sharpe_ratio = np.sqrt(252)*np.mean(excess_ret)/np.std(excess_ret)
    return sharpe_ratio

def calculate_max_DD(rets):
    rets = np.array(rets)
    cum_ret = np.cumprod(1+rets)-1
    #plt.plot(cum_ret)

    #HIGH WATERMARK -> MAXIMUM CUMULATIVE RET OF THE STRATEGY UNTIL TODAY
    high_watermark = np.zeros(cum_ret.shape[0])
    
    #DRAWDOWN IS THE DIFFERENCE BETWEEN CURRENT CUMRET AND THE HIGH WATERMARK
    #DRAWDOWN DURATION IS THE DISTANCE FROM TODAY TO THE HIGH WATERMARK DAY
    drawdown_duration = np.zeros(cum_ret.shape[0])
    
    drawdown = np.zeros(cum_ret.shape[0])

    for t in np.arange(1, cum_ret.shape[0]):
        high_watermark[t] = np.maximum(high_watermark[t-1], cum_ret[t])

        drawdown[t] = (1+cum_ret[t])/(1+high_watermark[t])-1
        if drawdown[t] == 0:
            drawdown_duration[t] = 0
        else:
            drawdown_duration[t] = drawdown_duration[t-1]+1
    
    max_dd, i = np.min(drawdown), np.argmin(drawdown)
    
    max_ddd = np.max(drawdown_duration)
    #return -> maximum drawdown value, maximum drawdown duration, index of max drawdown
    return max_dd, max_ddd, i