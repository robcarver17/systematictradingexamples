"""
Trading rules

Note that the scripted versions of these can be found in the carry and ewmac files

"""

import pandas as pd
import numpy as np
from common import cap_series, pd_readcsv, find_datediff, ROOT_DAYS_IN_YEAR, ewmac_forecast_scalar

def calc_carry_forecast(carrydata, price, f_scalar=30.0):

    """
    Carry calculation
    
    Formulation here will work whether we are trading the nearest contract or not
    
    For other asset classes you will have to work out nerpu (net expected return in price units) yourself
    """
    
    nerpu=carrydata.apply(find_datediff, axis=1)
    
    stdev_returns=volatility(price)
    ann_stdev=stdev_returns*ROOT_DAYS_IN_YEAR
    
    raw_carry=nerpu/ann_stdev
    
    forecast=raw_carry*f_scalar
    
    cap_forecast=cap_series(forecast)
    
    return cap_forecast

def volatility(price, vol_lookback=25):
    return pd.ewmstd(price - price.shift(1), span=vol_lookback, min_periods=vol_lookback)    


def calc_ewmac_forecast(price, Lfast, Lslow=None, usescalar=True):
    
    
    """
    Calculate the ewmac trading fule forecast, given a price and EWMA speeds Lfast, Lslow and vol_lookback
    
    Assumes that 'price' is daily data
    """
    ## price: This is the stitched price series
    ## We can't use the price of the contract we're trading, or the volatility will be jumpy
    ## And we'll miss out on the rolldown. See http://qoppac.blogspot.co.uk/2015/05/systems-building-futures-rolling.html

    if Lslow is None:
        Lslow=4*Lfast
    
    ## We don't need to calculate the decay parameter, just use the span directly
    
    fast_ewma=pd.ewma(price, span=Lfast)
    slow_ewma=pd.ewma(price, span=Lslow)
    raw_ewmac=fast_ewma - slow_ewma
    
    ## volatility adjustment
    stdev_returns=volatility(price)    
    vol_adj_ewmac=raw_ewmac/stdev_returns
    
    ## scaling adjustment
    if usescalar:
        f_scalar=ewmac_forecast_scalar(Lfast, Lslow)
        forecast=vol_adj_ewmac*f_scalar
    else:
        forecast=vol_adj_ewmac
    
    cap_forecast=cap_series(forecast, capmin=-20.0,capmax=20.0)
    
    return cap_forecast