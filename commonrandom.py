"""
Functions used to create random data
"""

from random import gauss
import numpy as np
import pandas as pd
import scipy.stats as st
import math as maths
from common import DAYS_IN_YEAR, ROOT_DAYS_IN_YEAR

def generate_trends(Nlength, Tlength , Xamplitude):
    """
    Generates a price process, Nlength returns, underlying trend with length T and amplitude X
    
    returns a vector of numbers as a list
    
    """

    halfAmplitude=Xamplitude/2.0
    trend_step=Xamplitude/Tlength
    
    cycles=int(np.ceil(Nlength/Tlength))
    
    trendup=list(np.arange(start=-halfAmplitude, stop=halfAmplitude, step=trend_step))
    trenddown=list(np.arange(start=halfAmplitude, stop=-halfAmplitude, step=-trend_step))
    alltrends=[trendup+trenddown]*int(np.ceil(cycles))
    alltrends=sum(alltrends, [])
    alltrends=alltrends[:Nlength]
    
    return alltrends

    
def generate_trendy_price(Nlength, Tlength , Xamplitude, Volscale):
    """
    Generates a trend of length N amplitude X, plus gaussian noise mean zero std. dev (vol scale * amplitude)
    
    returns a vector of numbers
    """
    
    stdev=Volscale*Xamplitude
    noise=generate_noise(Nlength, stdev)

    ## Can use a different process here if desired
    process=generate_trends(Nlength, Tlength , Xamplitude)    
    
    combined_price=[noise_item+process_item for (noise_item, process_item) in zip(noise, process)]
    
    return combined_price

def generate_noise(Nlength, stdev):
    """
    Generates a series of gaussian noise as a list Nlength
    """
    
    return [gauss(0.0, stdev) for Unused in range(Nlength)]


def arbitrary_timeseries(datalist, index_start=pd.datetime(2000,1,1)):
    """
    For nice plotting, convert a list of prices or returns into an arbitrary pandas time series
    """    
    
    ans=pd.TimeSeries(datalist, index=pd.date_range(start=index_start, periods=len(datalist), freq="BDay"))
    
    return ans

def threeassetportfolio(plength=5000, SRlist=[1.0, 1.0, 1.0], annual_vol=.15, clist=[.0,.0,.0], index_start=pd.datetime(2000,1,1)):

    (c1, c2, c3)=clist
    dindex=pd.date_range(start=index_start, periods=plength, freq="Bday")

    daily_vol=annual_vol/16.0
    means=[x*annual_vol/250.0 for x in SRlist]
    stds = np.diagflat([daily_vol]*3)
    corr=np.array([[1.0, c1, c2], [c1, 1.0, c3], [c2, c3, 1.0]])
 
    covs=np.dot(stds, np.dot(corr, stds))
    plength=len(dindex)

    m = np.random.multivariate_normal(means, covs, plength).T

    portreturns=pd.DataFrame(dict(one=m[0], two=m[1], three=m[2]), dindex)
    portreturns=portreturns[['one', 'two', 'three']]
    
    return portreturns


def skew_returns_annualised(annualSR=1.0, want_skew=0.0, voltarget=0.20, size=10000):
    annual_rets=annualSR*voltarget
    daily_rets=annual_rets/DAYS_IN_YEAR
    daily_vol=voltarget/ROOT_DAYS_IN_YEAR
    
    return skew_returns(want_mean=daily_rets,  want_stdev=daily_vol,want_skew=want_skew, size=10000)

def skew_returns(want_mean,  want_stdev, want_skew, size=10000):
    
    EPSILON=0.0000001
    shapeparam=(2/(EPSILON+abs(want_skew)))**2
    scaleparam=want_stdev/(shapeparam)**.5
    
    sample = list(np.random.gamma(shapeparam, scaleparam, size=size))
    
    if want_skew<0.0:
        signadj=-1.0
    else:
        signadj=1.0
    
    natural_mean=shapeparam*scaleparam*signadj
    mean_adjustment=want_mean - natural_mean 

    sample=[(x*signadj)+mean_adjustment for x in sample]
    
    return sample

