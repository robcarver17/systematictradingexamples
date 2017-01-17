
import Image
from random import gauss
import numpy as np
from matplotlib.pyplot import plot, show, xticks, xlabel, ylabel, legend, yscale, title, savefig, rcParams, figure, hist, scatter

import matplotlib.pylab as plt
from itertools import cycle
import pickle
import pandas as pd

lines = ["--","-","-."]
linecycler = cycle(lines)    

def twocorrelatedseries(dindex, daily_mean, daily_mean2, daily_vol, corr):

    means = [daily_mean, daily_mean2]  
    stds = [daily_vol]*2
    covs = [[stds[0]**2          , stds[0]*stds[1]*corr], 
            [stds[0]*stds[1]*corr,           stds[1]**2]] 

    no_periods=len(dindex)

    m = np.random.multivariate_normal(means, covs, no_periods).T

    data1=pd.TimeSeries(m[0], dindex)
    data2=pd.TimeSeries(m[1], dindex)
    diff=12*(np.mean(data1)- np.mean(data2)) - 0.01 ## account for costs
    
    SR1=(12**.5)*data1.mean()/data1.std()
    SR2=(12**.5)*data2.mean()/data1.std()
    diffS=SR1 - SR2
    
    return (diff, diffS) 





## path to difference for one thing no correlation
annual_vol=0.20
monthly_vol=annual_vol/12**.5

annual_mean1=(annual_vol*.3) 
monthly_mean1=annual_mean1/12

annual_mean2=annual_vol*.3
monthly_mean2=annual_mean2/12


## a 10% one sided tests runs at around 1.28 standard deviations
magic_number=1.28

## Make sure these match!

monte_carlos=5000



corrlist=[ 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95]
#corrlist=[-1.0, 0.0, 0.95]

year_list=[1,2,3, 5,7, 10,20, 50]

for year_length in year_list:
    for corr in corrlist:

        no_periods=year_length*12
        dindex=pd.date_range(pd.datetime(1980,1,1), periods=no_periods, freq="M")

        roll_list=[twocorrelatedseries(dindex, monthly_mean1, monthly_mean2, monthly_vol, corr=corr) for ii in range(monte_carlos)]
        means=[x[0] for x in roll_list]
        SR=[x[1] for x in roll_list]
        
        print "For %d years; correlation %.2f the SR figure is %.4f and mean figure is %.4f" % (year_length, 
                                                    corr, np.std(SR)*magic_number, np.std(means)*magic_number)
        
        mean_magic= np.std(means)*magic_number
        count=sum([x>mean_magic for x in means])
        
        print "Out of 100 managers %f were good" % (float(count)*(100.0/monte_carlos))
        