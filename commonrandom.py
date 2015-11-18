"""
Functions used to create random data
"""

from random import gauss
import numpy as np
import pandas as pd
from common import DAYS_IN_YEAR, ROOT_DAYS_IN_YEAR, arbitrary_timeindex
import scipy.signal as sg

def generate_siney_trends(Nlength, Tlength , Xamplitude):
    """
    Generates a price process, Nlength returns, underlying trend with length T and amplitude X
    as a sine wave
    
    returns a vector of numbers as a list
    
    """

    halfAmplitude=Xamplitude/2.0

    cycles=Nlength/Tlength
    cycles_as_pi=cycles*np.pi
    increment=cycles_as_pi/Nlength
    
    alltrends=[np.sin(x)*halfAmplitude for x in np.arange(0.0, cycles_as_pi, increment)]
    alltrends=alltrends[:Nlength]
    
    return alltrends
    

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

    

    
def generate_trendy_price(Nlength, Tlength , Xamplitude, Volscale, sines=False):
    """
    Generates a trend of length N amplitude X, plus gaussian noise mean zero std. dev (vol scale * amplitude)
    
    If sines=True then generates as a sine wave, otherwise straight line
    
    returns a vector of numbers
    """
    
    stdev=Volscale*Xamplitude
    noise=generate_noise(Nlength, stdev)

    ## Can use a different process here if desired
    if sines:
        process=generate_siney_trends(Nlength, Tlength , Xamplitude) 
    else:
        process=generate_trends(Nlength, Tlength , Xamplitude)    
    
    combined_price=[noise_item+process_item for (noise_item, process_item) in zip(noise, process)]
    
    return combined_price

def generate_noise(Nlength, stdev):
    """
    Generates a series of gaussian noise as a list Nlength
    """
    
    return [gauss(0.0, stdev) for Unused in range(Nlength)]




def threeassetportfolio(plength=5000, SRlist=[1.0, 1.0, 1.0], annual_vol=.15, clist=[.0,.0,.0], index_start=pd.datetime(2000,1,1)):

    (c1, c2, c3)=clist
    dindex=arbitrary_timeindex(plength, index_start)

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
    
    return skew_returns(want_mean=daily_rets,  want_stdev=daily_vol,want_skew=want_skew, size=size)

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

def autocorr_skewed_returns(rho, want_mean,  want_stdev, want_skew, size=10000):
    
    
    ## closed form correction for ar1 process noise
    noise_stdev=(want_stdev**2 * (1-rho))**.5
    noise_terms=skew_returns(want_mean, noise_stdev, want_skew, size)
    
    ## combine the noise with a filter
    return sg.lfilter((1,),(1,-rho),noise_terms)




def adj_moments_for_rho(want_rho, want_mean, want_skew, want_stdev):
    """
    Autocorrelation introduces biases into other moments of a distribution
    
    Here I correct for these
    """
    assert abs(want_rho)<=0.8
    
    mean_correction=1/(1-want_rho)
    
    if want_rho>=0.0:
        skew_correction=(1-want_rho)**.5
    else:
        skew_correction=np.interp(want_rho, [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -.2, -0.1],
                                            [.14,  .27,  .42,   .58,  .72, .84,  .93, .98 ])
    
    ## somewhat hacky, but we do a correction inside the random generation function already
    
    stdev_correction=np.interp(want_rho, [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -.2, -0.1,  
                                          0.0,.1,.2,.3,.4,.5,.6,.7,.8],
                               [2.24, 1.83, 1.58, 1.41, 1.29, 1.19, 1.12, 1.05,
                                1.0, .95,.91,.88 ,.85, .82 , .79,.77 ,.75])
    
    adj_want_stdev=want_stdev/stdev_correction
    adj_want_mean=want_mean/mean_correction
    adj_want_skew=want_skew/skew_correction

    return (adj_want_mean, adj_want_skew, adj_want_stdev)