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

def finalpr(x):
    if type(x) is pd.core.frame.DataFrame:
        return x.apply(geomean, 0)
    prod_returns=x+1.0
    cum_prod_returns=prod_returns.cumprod()
    final_pr=cum_prod_returns.values[-1]

    return final_pr

def geomean(x):
    final_pr=finalpr(x)
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


data=pd_readcsv("/home/rob/workspace/systematictradingexamples/plots_for_perhaps/USrealreturnsADJ.csv", "year")

bond_weight, equity_weight=[0.68, 0.32]

## resampled version
period_length=21
monte_runs=5000

results=[[1000.0, 1000.0, 1000.0]]
for period in range(period_length)[1:]:
    print(period)
    periodres=[]
    for monte in range(monte_runs):
        choose=[int(random()*len(data.index)) for notused in range(period)]
        subdata=data.iloc[choose,]
        subdata.index=pd.date_range(pd.datetime(1990,1,1), periods=period, freq="A")
        
        portfolio_returns=subdata.SP500*equity_weight+subdata.US10*bond_weight
        gmeans=finalpr(portfolio_returns)
        periodres.append(gmeans)

    med=np.percentile(periodres,50)*1000
    m5=np.percentile(periodres,5)*1000
    m95=np.percentile(periodres,95)*1000
    
    results.append([m5,med,m95])

results=np.array(results)
results=pd.DataFrame(results, pd.date_range(pd.datetime(2017,1,1), periods=period_length, freq="A"))
results.columns=["Lower 5%", "Median", "Upper 5%"]

results.iloc[:,0].plot(style="k-.")
results.iloc[:,1].plot(style="k-")
results.iloc[:,2].plot(style="k-.")


from optimisation import *
from matplotlib.pyplot import plot, show, xticks, xlabel, ylabel, legend, yscale, title, savefig, rcParams, figure, hist, text, bar, subplots
import matplotlib.pyplot as plt
import Image

def file_process(filename):
    fig = plt.gcf()
    fig.set_size_inches(18.5,10.5)
    fig.savefig("/home/rob/%s.png" % filename,dpi=300)
    fig.savefig("/home/rob/%sLOWRES.png" % filename,dpi=50)
    
    Image.open("/home/rob/%s.png" % filename).convert('L').save("/home/rob/%s.jpg" % filename)
    Image.open("/home/rob/%sLOWRES.png" % filename).convert('L').save("/home/rob/%sLOWRES.jpg" % filename)


frame=plt.gca()

frame.set_yticks([1000.0,  3000., 5000.])



rcParams.update({'font.size': 18})

file_process("fanchart")

show()
