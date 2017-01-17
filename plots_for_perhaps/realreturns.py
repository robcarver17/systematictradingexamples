import pandas as pd
import numpy as np
from random import random

def pd_readcsv(filename, date_index_name="DATETIME"):
    """
    Reads a pandas data frame, with time index labelled
    package_name(/path1/path2.., filename

    :param filename: Filename with extension
    :type filename: str

    :param date_index_name: Column name of date index
    :type date_index_name: list of str


    :returns: pd.DataFrame

    """

    ans = pd.read_csv(filename)
    #ans.index = pd.to_datetime(ans[date_index_name]).values
    ans.index = ans[date_index_name].values

    del ans[date_index_name]

    ans.index.name = None

    return ans

def geomean(x):
    if type(x) is pd.core.frame.DataFrame:
        return x.apply(geomean, 0)
    prod_returns=x+1.0
    cum_prod_returns=prod_returns.cumprod()
    final_pr=cum_prod_returns.values[-1]
    avg_pr=final_pr**(1/float(len(x.index)))
    return avg_pr-1.0

def geostd(x):
    if type(x) is pd.core.frame.DataFrame:
        return x.apply(geostd, 0)
    gm=geomean(x)
    
    def _gsone(xitem, gm):
        return (np.log((1+xitem)/(1+gm)))**2.0
        
    items=[_gsone(xitem, gm) for xitem in x.values]
    
    return np.exp((np.mean(items))**.5)-1.0


data=pd_readcsv("/home/rob/workspace/systematictradingexamples/plots_for_perhaps/USrealreturns.csv", "year")

## resampled version
period_length=50
monte_runs=1000

eqbeatbonds=0
eqbeatcash=0
bobeatcash=0
eqbeatport=0

for monte in range(monte_runs):
    choose=[int(random()*len(data.index)) for notused in range(period_length)]
    subdata=data.iloc[choose,]
    subdata.index=pd.date_range(pd.datetime(1990,1,1), periods=period_length, freq="A")
    portfolio=subdata.SP500*.6+subdata.US10*.4
    
    gmeans=geomean(subdata)
    if gmeans.SP500>gmeans.US10:
        eqbeatbonds=eqbeatbonds+1
    if gmeans.SP500>gmeans.TBILL:
        eqbeatcash=eqbeatcash+1
    if gmeans.US10>gmeans.TBILL:
        bobeatcash=bobeatcash+1
        
    if gmeans.SP500>geomean(portfolio):
        eqbeatport=eqbeatport+1
        
## portfolio optimisation
wts=[0.4, 0.6]
wts=wts+[1-sum(wts)]

portfolio=data.SP500*wts[0]+data.US10*wts[1]+data.TBILL*wts[2]

print(geomean(portfolio))
print(geostd(portfolio))
print((geomean(portfolio) - 0.005)/geostd(portfolio))

