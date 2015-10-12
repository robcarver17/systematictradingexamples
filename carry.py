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
from common import cap_series, pd_readcsv, find_datediff, get_price_for_instrument, get_carry_data, ROOT_DAYS_IN_YEAR


## This is the stitched price series
## We can't use the price of the contract we're trading, or the volatility will be jumpy
code="CRUDE_W"
price=get_price_for_instrument(code)


"""
Formulation here will work whether we are trading the nearest contract or not

For other asset classes you will have to work out nerpu (net expected return in price units) yourself
"""

data=get_carry_data(code)
data[['TRADED','NEARER']].plot()
plt.show()



#d1=pd.datetime(2007,1,1)
#d2=pd.datetime(2009,12,31)


nerpu=data.apply(find_datediff, axis=1)
nerpu.plot()
plt.title("Nerpu")
plt.show()

## Shouldn't need changing
vol_lookback=25
stdev_returns=pd.ewmstd(price - price.shift(1), span=vol_lookback)
ann_stdev=stdev_returns*ROOT_DAYS_IN_YEAR

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

