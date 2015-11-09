"""
Check the autocorrelation (and other) properties of a couple of different trading strategies



I'm using my usual set of around 40 futures markets, with data from the mid 1970's onwards. Note that although only a few markets have that much history). 

First I run the relevant trading rule variation on each instrument. I then generate the relevant statistic for each equity curve, and plot the histogram of these statistics.


This method will ignore the fact that different instruments have varying history. So the other thing I do is to first split each instrument's equity curve into annual 'bins'. I then generate a statistic for each year of each instrument. Finally again I show a histogram of these statistics. This allows us to see how variable the statistics are over time, and also upweights assets with more history.


I'm going to focus on the annualised standard deviation, skew and return autocorrelation over different periods.

 

Finally to avoid filling this post with thousands of pictures... okay 168 pictures* I'm going to report just the average of each statistic rather than plot all the distributions.

 

* seven trading rule variations, two methods, four statistics, 3 time periods



"""

from tradingrules import calc_ewmac_forecast, volatility, calc_carry_forecast
from common import get_price_for_instrument, get_list_code,  get_point_sizes, ROOT_DAYS_IN_YEAR, calculate_pandl,  get_carry_data, stack_ts, break_up_ts, account_curve, remove_nans_from_list
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plotstuff=False

## needed to calculate p&l
pointsizedict=get_point_sizes()

## Doesn't matter what this is
accountvalue=1000
risktarget=0.20
ann_cash_vol_target=risktarget*accountvalue

## we're not going to worry about costs, position inertia or rounding in this simple example
daily_cash_vol_target=ann_cash_vol_target/ROOT_DAYS_IN_YEAR


code_list=get_list_code()
priceStack=[]
currentpriceStack=[]
volStack=[]
carrydataStack=[]

for code in code_list:
    price=get_price_for_instrument(code)
    carrydata=get_carry_data(code)
    current_price=carrydata.TRADED

    priceStack.append(price)
    currentpriceStack.append(current_price)
    carrydataStack.append(carrydata)
    
    stdev=volatility(price)
    price_volatility=100*stdev/current_price
    volStack.append(price_volatility)


LfastList=[None, 2, 4, 8, 16, 32, 64]

## If you set Lfast to None, will use carry rule
for Lfast in LfastList:
    if Lfast is None:
        variation_name="carry"
    else:
        variation_name="EWMAC %d,%d" % (Lfast, Lfast*4)

    ### Generate all the forecasts
    fcastStack=[]
    
    for codeidx in range(len(code_list)):
        
        price=priceStack[codeidx]
        carrydata=carrydataStack[codeidx]
        current_price=currentpriceStack[codeidx]
    
        if Lfast is None:
            forecast=calc_carry_forecast(carrydata, price)
            ## carry
        else:
            forecast=calc_ewmac_forecast(price, Lfast)
        
        fcastStack.append(forecast)
        
    fcastStack_all=pd.concat(fcastStack, axis=0).values
        
    ## Generate our p&l
    
    pandlstack=[]
    
    for codeidx in range(len(code_list)):
        code=code_list[codeidx]
        price_volatility=volStack[codeidx]
    
        ## really low vol screws up position calculation; we'd never 
        #price_volatility=cap_series(price_volatility, capmin=0.01)
        
        price=priceStack[codeidx]
        current_price=currentpriceStack[codeidx]
        forecast=fcastStack[codeidx]
        
        ##  block value = value of a 1% move
        block_value=current_price*0.01*pointsizedict[code]
        
        instr_cccy_volatility=block_value*price_volatility
        ## note we're not doing a portfolio here, so don't need currency
        instr_value_volatility=instr_cccy_volatility
            
        vol_scalar=daily_cash_vol_target/instr_value_volatility
            
        subsys_position_natural=forecast*vol_scalar/10.0
        
        p1=calculate_pandl(subsys_position_natural, price, pointsizedict[code])/accountvalue
                    
        pandlstack.append(account_curve(p1.iloc[:,0]))
    
    
    """
    
    How shall we evaluate this stuff
    
    - look at the distribution by code
    - look at the distribution by 1 year blocks (will upweight codes with more data)
    
    """
    
    ## create a string of one year blocks
    blockedout=[break_up_ts(x) for x in pandlstack]
    blockedout=sum(blockedout, [])
    blockedout=[account_curve(x) for x in blockedout]
    
    
    """
    Calculate statistics and print
    """
    
    titlelist=[ "stdev","skew", "autocorr"]
    freqlist=["Daily", "Weekly", "Monthly"]

    for freq in freqlist:
        ## Resample
        ## Assumes business day initially
        pandlstack_atfreq=[x.new_freq(freq) for x in pandlstack]
        blockedout_atfreq=[x.new_freq(freq) for x in blockedout]
        
        print 
        
        for title in titlelist:
            if title=="stdev":
                results=[x.std() for x in pandlstack_atfreq]
                results2=[x.std() for x in blockedout_atfreq]
            elif title=="skew":
                results=[x.skew() for x in pandlstack_atfreq]
                results2=[x.skew() for x in blockedout_atfreq]
            elif title=="autocorr":
                results=[x.autocorr() for x in pandlstack_atfreq]
                results2=[x.autocorr() for x in blockedout_atfreq]
            else:
                print "title %s not recognised" % title
            
            results=remove_nans_from_list(results)
            
            tstat=np.mean(results)/np.std(results)
            
            fulltitle="%s %s %s by code: average %.4f (%.4f)" % (variation_name, freq, title, np.mean(results), tstat)
            
            if plotstuff:
                plt.hist(results)
                plt.title(fulltitle)
                plt.show()
            print fulltitle
            
            
            results2=remove_nans_from_list(results2)
            ## will be approx 40 x 10 observations
            tstat=np.mean(results2)/np.std(results2)

            fulltitle="%s %s %s by code and year: average %.4f (%.4f)" % (variation_name, freq, title, np.mean(results2), tstat)
        
            if plotstuff:
                plt.hist(results2)
                plt.title(fulltitle)
                plt.show()
            print fulltitle
        
        