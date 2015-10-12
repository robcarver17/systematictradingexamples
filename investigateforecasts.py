"""
Here we'll check the hypothesis that we should use continous, rather than binary, forecasts

"""

"""
First job - collect a stack of pooled data

We're going to be plotting the forecast value versus the realised normalised return

"""

from tradingrules import calc_ewmac_forecast, volatility, calc_carry_forecast
from common import get_price_for_instrument, get_list_code, cap_series, get_point_sizes, ROOT_DAYS_IN_YEAR, calculate_pandl, sharpe, get_carry_data, stack_ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

def make_binary(xseries, rounded=False):
    """
    Turn a forecast into a binary version +10, -10
    
    We do this in two ways:
    
    - true binary
    - rounded binary. If abs(x)<5, forecast is zero. Else forecast is 
    """
    
    if rounded:
        (min_value, max_value)=(-5.0, 5.0)
    else:
        (min_value, max_value)=(0.0, 0.0)

    return xseries.apply(make_binary_row,  args=(min_value, max_value))
    

def make_binary_row(x, min_value,max_value):
    """
    Binarise forecast
    """

    if x<min_value:
        return -10.0
    elif x>max_value:
        return 10.0
    
    return 0.0



## If you set Lfast to None, will use carry rule
Lfast=None


code_list=get_list_code()

priceStack=[]
currentpriceStack=[]
fcastStack=[]
binary1Stack=[]
binary2Stack=[]
vnStack=[]
volStack=[]
root_days_in_week=5**.5

for code in code_list:
    price=get_price_for_instrument(code)
    carrydata=get_carry_data(code)
    current_price=carrydata.TRADED

    if Lfast is None:
        forecast=calc_carry_forecast(carrydata, price)
        ## carry
    else:
        forecast=calc_ewmac_forecast(price, Lfast)
    
    stdev=volatility(price)*root_days_in_week
    price_volatility=100*stdev/current_price
    next_weeks_return =price.shift(-5) - price
    next_weeks_vol_norm_return =next_weeks_return/stdev
    next_weeks_vol_norm_return=cap_series(next_weeks_vol_norm_return, capmin=-50.0,capmax=50.0)

    fcastStack.append(forecast)
    vnStack.append(next_weeks_vol_norm_return)
    
    binary1forecast=make_binary(forecast, rounded=False)
    binary2forecast=make_binary(forecast, rounded=True)

    binary1Stack.append(binary1forecast)
    binary2Stack.append(binary2forecast)
    
    volStack.append(price_volatility)
    
    priceStack.append(price)
    currentpriceStack.append(current_price)
    
fcastStack_all=pd.concat(fcastStack, axis=0).values
vnStack_all=pd.concat(vnStack, axis=0).values

conditionals=[]
fcastpoints=[-20.0,  -10.0,  0.0,  10.0,  20.0]

for idx in range(len(fcastpoints))[1:]:
    condp=[vnStack_all[i] for i in range(len(vnStack_all)) if fcastStack_all[i]>=fcastpoints[idx-1] and fcastStack_all[i]<fcastpoints[idx]]
    condp=[x for x in condp if not np.isnan(x)]
    conditionals.append(condp)
    

for idx in range(len(conditionals)):
    print "Bin %d to %d: mean %.1f" % (int(fcastpoints[idx]), int(fcastpoints[idx+1]), np.mean(conditionals[idx])*100)


for idx in range(len(conditionals))[1:]:
    ans=ttest_ind(conditionals[idx-1], conditionals[idx])
    print "Bin %d to %d vs %d to %d" % (int(fcastpoints[idx-1]), int(fcastpoints[idx]), int(fcastpoints[idx]), int(fcastpoints[idx+1]))
    print "Paired t-test :  %.2f %.2f" % ( ans[0], ans[1])

#### Okay ... what about sharpe ratio?

pointsizedict=get_point_sizes()


## Doesn't matter what this is
accountvalue=1000
risktarget=0.25
ann_cash_vol_target=risktarget*accountvalue

## we're not going to worry about costs, position inertia or rounding in this simple example
daily_cash_vol_target=ann_cash_vol_target/ROOT_DAYS_IN_YEAR

natural_pandlstack=[]
b1_pandlstack=[]
b2_pandlstack=[]

for codeidx in range(len(code_list)):
    code=code_list[codeidx]
    price_volatility=volStack[codeidx]

    ## really low vol screws up position calculation; we'd never 
    #price_volatility=cap_series(price_volatility, capmin=0.01)
    
    price=priceStack[codeidx]
    current_price=currentpriceStack[codeidx]
    forecast=fcastStack[codeidx]
    forecast_binary1=binary1Stack[codeidx]
    forecast_binary2=binary2Stack[codeidx]
    
    ##  block value = value of a 1% move
    block_value=current_price*0.01*pointsizedict[code]
    
    instr_cccy_volatility=block_value*price_volatility
    ## note we're not doing a portfolio here, so don't need currency
    instr_value_volatility=instr_cccy_volatility
        
    vol_scalar=daily_cash_vol_target/instr_value_volatility
        
    subsys_position_natural=forecast*vol_scalar/10.0
    
    subsys_position_binary1=forecast_binary1*vol_scalar/10.0
    subsys_position_binary2=forecast_binary2*vol_scalar/10.0
    
    p1=calculate_pandl(subsys_position_natural, price, pointsizedict[code])
    p2=calculate_pandl(subsys_position_binary1, price, pointsizedict[code])
    p3=calculate_pandl(subsys_position_binary2, price, pointsizedict[code])
                
    sr1=sharpe(p1)
    sr2=sharpe(p2)
    sr3=sharpe(p3)

    print "%s: Sharpe ratios Natural %.2f Binary1 %.2f Binary 2 %.2f" % (code, sr1, sr2, sr3)
    print "    Vols  Natural %.2f Binary1 %.2f Binary 2 %.2f" % (p1.std()/daily_cash_vol_target, p2.std()/daily_cash_vol_target, p3.std()/daily_cash_vol_target)

    natural_pandlstack.append(p1)
    b1_pandlstack.append(p2)
    b2_pandlstack.append(p3)

"""
FIX ME

THIS IS A ROUGH PORTFOLIO

EITHIER DO PROPER PORTFOLIO OR STACK IN TIME
"""


natural_pandlstack=stack_ts(natural_pandlstack)
b1_pandlstack=stack_ts(b1_pandlstack)
b2_pandlstack=stack_ts(b2_pandlstack)

natural_pandlstack.cumsum().ffill().plot(label="Natural")
b1_pandlstack.cumsum().ffill().plot(label="Binary 1")
b2_pandlstack.cumsum().ffill().plot("Binary 2")
plt.legend()
plt.show()

x1=natural_pandlstack.cumsum() - b1_pandlstack.cumsum() 
x1.ffill().plot()
plt.show()

x2=natural_pandlstack.cumsum() - b2_pandlstack.cumsum() 
x2.ffill().plot()
plt.show()



sr1=sharpe(natural_pandlstack)
sr2=sharpe(b1_pandlstack)
sr3=sharpe(b2_pandlstack)

print ""
print "ALL Sharpe ratios Natural %.2f Binary1 %.2f Binary 2 %.2f" % (sr1, sr2, sr3)
print "    Vols  Natural %.2f Binary1 %.2f Binary 2 %.2f" % (natural_pandlstack.std()/daily_cash_vol_target, b1_pandlstack.std()/daily_cash_vol_target, b2_pandlstack.std()/daily_cash_vol_target)
