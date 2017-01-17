"""
test the 12 month asset allocation model

no costs currently
"""

import pandas as pd
import numpy as np
from datetime import datetime as dt

def file_process(filename):
    fig = plt.gcf()
    fig.set_size_inches(18.5,10.5)
    fig.savefig("/home/rob/%s.png" % filename,dpi=300)
    fig.savefig("/home/rob/%sLOWRES.png" % filename,dpi=50)
    
    Image.open("/home/rob/%s.png" % filename).convert('L').save("/home/rob/%s.jpg" % filename)
    Image.open("/home/rob/%sLOWRES.png" % filename).convert('L').save("/home/rob/%sLOWRES.jpg" % filename)

"""
data=pd.read_csv("/home/rob/workspace/systematictradingexamples/plots_for_perhaps/USmonthlyreturns.csv")
data.index=data.Date

vols=[0.15, 0.08]

risk_weights=[.5, .5]

"""

SRdifftable=[     -0.25, -0.2, -0.15,  -0.1, -0.05, 0.0, 0.05, 0.1,  0.15, 0.2, 0.25]
multipliers=[.6,    .66, .77,     .85,  .94,   1.0, 1.11, 1.19, 1.3, 1.37, 1.48]


def read_ts_csv(fname, dindex="Date"):
    data=pd.read_csv(fname)
    dateindex=[dt.strptime(dx, "%d/%m/%y") for dx in list(data[dindex])]
    data.index=dateindex
    del(data[dindex])
    
    return data


rawdata=read_ts_csv("/home/rob/workspace/systematictradingexamples/plots_for_perhaps/MSCI_data.csv")
refdata=pd.read_csv("/home/rob/workspace/systematictradingexamples/plots_for_perhaps/MSCI_ref.csv")

def sign(x):
    if x is None:
        return None
    if np.isnan(x):
        return np.NAN
    if x==0:
        return 0.0
    if x<0:
        return -1.0
    if x>0:
        return 1.0



def get_annual_tr(tickname, rawdata):
    total_returns=rawdata[tickname+"_TR"]
    pecrets=(total_returns / total_returns.shift(12)) - 1.0
    mperces=(total_returns.diff()/total_returns.shift(1)) - 1.0
    stdrets=mperces.std()*(12**.5)
    return pecrets/stdrets

def get_forward_tr(tickname, rawdata, months):
    total_returns=rawdata[tickname+"_TR"]
    pecrets=(total_returns.shift(-months) / total_returns) - 1.0
    mperces=(total_returns.diff()/total_returns.shift(1)) - 1.0
    stdrets=mperces.std()*(12**.5)
    return pecrets/stdrets
    


tickers=list(refdata[refdata.Type=="Country"].Country.values) #mom 17bp


def vl(emordev):
    if emordev=="EM":
        return .23
    else:
        return .15
    
vols=[vl(refdata[refdata.Country==ticker].EmorDEV.values[0]) for ticker in tickers]

def get_monthly_tr(tickname, rawdata):
    total_returns=rawdata[tickname+"_TR"]
    return (total_returns / total_returns.shift(1)) - 1.0

def calc_asset_returns(rawdata, tickers):
    asset_returns=pd.concat([get_monthly_tr(tickname, rawdata) for tickname in tickers], axis=1)
    asset_returns.columns=tickers

    return asset_returns

data=calc_asset_returns(rawdata, tickers)

conditioner=[get_annual_tr(tickner, rawdata) for tickner in tickers]
dependent=[get_forward_tr(tickner, rawdata, 3) for tickner in tickers]

## OR ...
#import random
#conditioner=[pd.Series([random.gauss(0.0, 1.0) for unused in ehup.index], ehup.index) for ehup in conditioner]





futreturns=[list(x.values) for x in dependent]
condvariable=[list(x.values) for x in conditioner]

futreturns=sum(futreturns, [])
condvariable=sum(condvariable, [])


cmean=np.nanmean(condvariable)
cstd=np.nanstd(condvariable)

condreturns_pair=[(fr,cv) for fr,cv in zip(futreturns, condvariable) if not np.isnan(fr) and not np.isnan(cv)]

upper_bound1=cmean+4*cstd
lower_bound1=cmean+cstd
condreturns=[fr for fr,cv in condreturns_pair if cv<upper_bound1 and cv>lower_bound1]
condreturns_cond=[cv for fr,cv in condreturns_pair if cv<=upper_bound1 and cv>lower_bound1]


upper_bound2=cmean+cstd
lower_bound2=cmean-cstd
condreturns2=[fr for fr,cv in condreturns_pair if cv<upper_bound2 and cv>lower_bound2]
condreturns_cond2=[cv for fr,cv in condreturns_pair if cv<=upper_bound2 and cv>lower_bound2]



upper_bound3=cmean-cstd
lower_bound3=cmean-4*cstd
condreturns3=[fr for fr,cv in condreturns_pair if cv<upper_bound3 and cv>lower_bound3]
condreturns_cond3=[cv for fr,cv in condreturns_pair if cv<=upper_bound3 and cv>lower_bound3]

def thing(retstrip):
    avg=np.nanmean(retstrip)
    std=np.nanstd(retstrip)
    length=len([x for x in retstrip if not np.isnan(x)])
    return (avg, std, length, std/(length**.5))

from optimisation import *
from matplotlib.pyplot import plot, show, xticks, xlabel, ylabel, legend, yscale, title, savefig, rcParams, figure, hist, text, bar, subplots
import matplotlib.pyplot as plt
import Image

from itertools import cycle

def linehist(x, color="blue", linestyle="-"):
    y,binEdges =np.histogram(x, bins=50)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    plot(bincenters,y,'-', color=color, linestyle=linestyle) 


lines = ["-","--"]
linecycler = cycle(lines)    
colorcycler=cycle(["red", "blue"])

linehist(condreturns, linestyle=next(linecycler), color=next(colorcycler))
linehist(condreturns2, linestyle=next(linecycler), color=next(colorcycler))




frame=plt.gca()

frame.get_yaxis().set_visible(False)
frame.set_xticks([-1.0, 0, 1.0])
frame.set_xlim([-2.0, 2.0])

legend(["Above %.2f" % lower_bound1,"Below %.2f" % upper_bound2])

rcParams.update({'font.size': 18})


file_process("conditionalreturn")



show()

