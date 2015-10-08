"""
This file shows how to calculate the Carry trading rule for crude oil futures

As in chapter 7 / appendix B of "Systematic Trading" by Robert Carver (www.systematictrading.org)

Required: pandas, matplotlib

USE AT YOUR OWN RISK! No warranty is provided or implied.

Handling of NAN's and Inf's isn't done here (except within pandas), 
And there is no error handling!

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


def pd_readcsv(filename):
    """
    Reads the pandas dataframe from a filename, given the index is correctly labelled
    """

    
    ans=pd.read_csv(filename)
    
    print ans
    
    ans.index=pd.to_datetime(ans['DATETIME'])
    del ans['DATETIME']
    ans.index.name=None
    
    return ans

filename="data/CRUDE_W_carrydata.csv"
data=pd_readcsv(filename)

## This is the stitched price series
## We can't use the price of the contract we're trading, or the volatility will be jumpy
filename="data/CRUDE_W_price.csv"
price=pd_readcsv(filename)


"""
Formulation here will work whether we are trading the nearest contract or not

For other asset classes you will have to work out nerpu (net expected return in price units) yourself
"""

data[['TRADED','NEARER']].plot()
plt.show()

def find_datediff(data_row):
    """
    data differential for a single row
    """
    if np.isnan(data_row.NEAR_MONTH) or np.isnan(data_row.TRADE_MONTH):
        return np.nan
    nearest_dt=pd.to_datetime(str(int(data_row.NEAR_MONTH)), format="%Y%m")
    trade_dt=pd.to_datetime(str(int(data_row.TRADE_MONTH)), format="%Y%m")
    
    distance = trade_dt - nearest_dt
    distance_years=distance.days/365.25
    
    price_diff=data_row.TRADED - data_row.NEARER
    
    return price_diff/distance_years


#d1=pd.datetime(2007,1,1)
#d2=pd.datetime(2009,12,31)


nerpu=data.apply(find_datediff, axis=1)
nerpu.plot()
plt.title("Nerpu")
plt.show()

## Shouldn't need changing
vol_lookback=25
stdev_returns=pd.ewmstd(price - price.shift(1), span=vol_lookback)
ann_stdev=stdev_returns*16.0

raw_carry=nerpu/ann_stdev
f_scalar=30.0

raw_carry.plot()
plt.title("Raw carry")
plt.show()

forecast=raw_carry*f_scalar

c_forecast=cap_series(forecast)

data_to_plot=pd.concat([forecast,c_forecast], axis=1)
data_to_plot.columns=['Forecast','Capped forecast']
data_to_plot.plot()
plt.show()

