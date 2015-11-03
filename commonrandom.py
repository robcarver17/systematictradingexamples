"""
Functions used to create random data
"""

from random import gauss
import numpy as np
import pandas as pd

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
    
    ans=pd.TimeSeries(datalist, index=pd.date_range(start=index_start, periods=len(datalist)))
    
    return ans

def threeassetportfolio(plength=5000, SRlist=[1.0, 1.0, 1.0], annual_vol=.15, clist=[.0,.0,.0]):

    (c1, c2, c3)=clist
    dindex=pd.date_range(start=pd.datetime(1990,1,1), periods=plength)

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

