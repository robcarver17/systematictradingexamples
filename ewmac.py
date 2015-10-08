"""
This file shows how to calculate the EWMAC trading rule for crude oil futures

As in chapter 7 / appendix B of "Systematic Trading" by Robert Carver (www.systematictrading.org)

Required: pandas, matplotlib

USE AT YOUR OWN RISK! No warranty is provided or implied.

Handling of NAN's and Inf's isn't done here (except within pandas), 
And there is no error handling!

"""

import pandas as pd
import matplotlib.pyplot as plt

def pd_readcsv(filename):
    """
    Reads the pandas dataframe from a filename, given the index is correctly labelled
    """

    
    ans=pd.read_csv(filename)
    ans.index=pd.to_datetime(ans['DATETIME'])
    del ans['DATETIME']
    ans.index.name=None
    
    return ans


def forecast_scalar(Lfast, Lslow):
    """
    Function to return the forecast scalar (table 49 of the book)
    
    Only defined for certain values
    """
    
    
    fsdict=dict(l2_8=10.6, l4_16=7.5, l8_32=5.3, l16_64=3.75, l32_128=2.65, l64_256=1.87)
    
    lkey="l%d_%d" % (Lfast, Lslow)
    
    if lkey in fsdict:
        return fsdict[lkey]
    else:
        print "Warning: No scalar defined for Lfast=%d, Lslow=%d, using default of 1.0" % (Lfast, Lslow)
        return 1.0

    
    
def cap_forecast(xrow, capmin,capmax):
    """
    Cap forecasts.
    
    """

    ## Assumes we have a single column    
    x=xrow[0]

    if x<capmin:
        return capmin
    elif x>capmax:
        return capmax
    
    return x

def cap_series(xseries, capmin=-20.0,capmax=20.0):
    """
    Apply capping to each element of a time series
    For a long only investor, replace -20.0 with 0.0

    """
    return xseries.apply(cap_forecast, axis=1, args=(capmin, capmax))

    
"""
get some data
"""

## This is the stitched price series
## We can't use the price of the contract we're trading, or the volatility will be jumpy
## And we'll miss out on the rolldown. See http://qoppac.blogspot.co.uk/2015/05/systems-building-futures-rolling.html

filename="data/CRUDE_W_price.csv"
price=pd_readcsv(filename)

## Shouldn't need changing
vol_lookback=25
    
"""
Calculate the ewmac trading fule forecast, given a price and EWMA speeds Lfast, Lslow and vol_lookback

Assumes that 'price' is daily data
"""

Lfast=16
Lslow=4*Lfast

d1=pd.datetime(2007,1,1)
d2=pd.datetime(2009,12,31)
    
## We don't need to calculate the decay parameter, just use the span directly

fast_ewma=pd.ewma(price, span=Lfast)
slow_ewma=pd.ewma(price, span=Lslow)
raw_ewmac=fast_ewma - slow_ewma

data_to_plot=pd.concat([price, fast_ewma, slow_ewma], axis=1)
data_to_plot.columns=['Price', 'Fast', 'Slow']

data_to_plot[d1:d2].plot()
plt.show()

raw_ewmac[d1:d2].plot()
plt.title("Raw EWMAC")
plt.show()

## volatility adjustment
stdev_returns=pd.ewmstd(price - price.shift(1), span=vol_lookback)    
vol_adj_ewmac=raw_ewmac/stdev_returns

vol_adj_ewmac[d1:d2].plot()
plt.title("Vol adjusted")
plt.show()

## scaling adjustment
f_scalar=forecast_scalar(Lfast, Lslow)

forecast=vol_adj_ewmac*f_scalar


cap_forecast=cap_series(forecast, capmin=-20.0,capmax=20.0)

data_to_plot=pd.concat([forecast, cap_forecast], axis=1)
data_to_plot.columns=['Scaled Forecast', 'Capped forecast']

data_to_plot[d1:d2].plot()
plt.show()



