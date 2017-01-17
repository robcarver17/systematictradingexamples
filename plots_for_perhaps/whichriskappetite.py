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


data=pd_readcsv("/home/rob/workspace/systematictradingexamples/plots_for_perhaps/USrealreturnsADJ.csv", "year")

portfolios=dict([("All Equities", [0.0,1.0]),
                 ("Maximum risk",[.2,.8]),
                 ("Compromise",[.4,.6]),
("Maximum SR",[.68,.32])                 
                 ])

## resampled version
period_length=5
monte_runs=100000

results=dict([("All Equities", []),
                 ("Maximum risk",[]),
                 ("Compromise",[]),
("Maximum SR",[])                 
                 ])
for monte in range(monte_runs):
    choose=[int(random()*len(data.index)) for notused in range(period_length)]
    subdata=data.iloc[choose,]
    subdata.index=pd.date_range(pd.datetime(1990,1,1), periods=period_length, freq="A")
    
    for resname in results.keys():
        bond_weight=portfolios[resname][0]
        equity_weight=portfolios[resname][1]
        portfolio_returns=subdata.SP500*equity_weight+subdata.US10*bond_weight
    
        gmeans=geomean(portfolio_returns)
        results[resname].append(gmeans)


from optimisation import *
from matplotlib.pyplot import plot, show, xticks, xlabel, ylabel, legend, yscale, title, savefig, rcParams, figure, hist, text, bar, subplots
import matplotlib.pyplot as plt
import Image

from itertools import cycle

from datetime import datetime as dt

def linehist(x, color="blue", linestyle="-"):
    y,binEdges =np.histogram(x, bins=50)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    plot(bincenters,y,'-', color=color, linestyle=linestyle)
    
    return list(y),list(bincenters) 

def revperc(x, a):

    return float(len([xx for xx in x if xx<=a]))/float(len(x))

def file_process(filename):
    fig = plt.gcf()
    fig.set_size_inches(18.5,10.5)
    fig.savefig("/home/rob/%s.png" % filename,dpi=300)
    fig.savefig("/home/rob/%sLOWRES.png" % filename,dpi=50)
    
    Image.open("/home/rob/%s.png" % filename).convert('L').save("/home/rob/%s.jpg" % filename)
    Image.open("/home/rob/%sLOWRES.png" % filename).convert('L').save("/home/rob/%sLOWRES.jpg" % filename)

def plot_vertical_line(x, y, binEdges, linestyles="dashed", colors="k"):
    plt.vlines(x, 0, np.interp(x, binEdges, y), linestyles=linestyles, colors=colors)

resname="All Equities"
ax1=plt.subplot(221)
y,binEdges=linehist(results[resname])

plt.setp(ax1.get_xticklabels(), visible=False)
print resname
print np.median(results[resname])
print np.mean(results[resname])
print np.percentile( results[resname],5)
print revperc(results[resname], 0.0)
title(resname)
plot_vertical_line(np.median(results[resname]), y, binEdges, linestyles="dashed")
plot_vertical_line(0.0, y, binEdges, linestyles="solid", colors="gray")

frame=plt.gca()


frame.get_yaxis().set_visible(False)


resname="Maximum risk"
ax2=plt.subplot(223, sharex=ax1, sharey=ax1)
y,binEdges=linehist(results[resname])
plt.setp(ax2.get_xticklabels(), fontsize=18)
print resname
print np.median(results[resname])
print np.percentile(results[resname],5)
print revperc(results[resname], 0.0)
title(resname)
plot_vertical_line(np.median(results[resname]), y, binEdges, linestyles="dashed")
plot_vertical_line(0.0, y, binEdges, linestyles="solid", colors="gray")

frame=plt.gca()

frame.get_yaxis().set_visible(False)


resname="Compromise"
ax3=plt.subplot(222, sharex=ax1, sharey=ax1)
y,binEdges=linehist(results[resname])

plt.setp(ax3.get_xticklabels(), visible=False)
print resname
print np.median(results[resname])
print np.percentile(results[resname], 5)
print revperc(results[resname], 0.0)
title(resname)
plot_vertical_line(np.median(results[resname]), y, binEdges, linestyles="dashed")
plot_vertical_line(0.0, y, binEdges, linestyles="solid", colors="gray")

frame=plt.gca()

frame.get_yaxis().set_visible(False)


resname="Maximum SR"
ax4=plt.subplot(224, sharex=ax1, sharey=ax1)
y,binEdges=linehist(results[resname])

title(resname)

plt.setp(ax4.get_xticklabels(), fontsize=18)
print resname
print np.median(results[resname])
print np.percentile(results[resname],5 )
print revperc(results[resname], 0.0)
plot_vertical_line(np.median(results[resname]), y, binEdges, linestyles="dashed")
plot_vertical_line(0.0, y, binEdges, linestyles="solid", colors="gray")


frame=plt.gca()

frame.get_yaxis().set_visible(False)
#frame.set_xticks([-0.05, 0.0, 0.05, 0.10, 0.15])
#frame.set_xlim([-0.1, 0.2])

frame.set_xticks([ -.1, 0.0,  0.10, 0.2, ])
frame.set_xlim([-0.2, 0.3])


rcParams.update({'font.size': 18})

file_process("riskappetite%d" % period_length)

show()
