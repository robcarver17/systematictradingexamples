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
from common import pd_readcsv, cap_series, ewmac_forecast_scalar, get_price_for_instrument

    
"""
get some data
"""

## This is the stitched price series
## We can't use the price of the contract we're trading, or the volatility will be jumpy
## And we'll miss out on the rolldown. See http://qoppac.blogspot.co.uk/2015/05/systems-building-futures-rolling.html
code="CRUDE_W"
price=get_price_for_instrument(code)


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
f_scalar=ewmac_forecast_scalar(Lfast, Lslow)

forecast=vol_adj_ewmac*f_scalar


cap_forecast=cap_series(forecast, capmin=-20.0,capmax=20.0)

data_to_plot=pd.concat([forecast, cap_forecast], axis=1)
data_to_plot.columns=['Scaled Forecast', 'Capped forecast']

data_to_plot[d1:d2].plot()
plt.show()



