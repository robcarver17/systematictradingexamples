
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

def twocorrelatedseries(no_periods, period_mean, period_mean2, period_vol, corr):

    means = [period_mean, period_mean2]  
    stds = [period_vol]*2
    covs = [[stds[0]**2          , stds[0]*stds[1]*corr], 
            [stds[0]*stds[1]*corr,           stds[1]**2]] 

    m = np.random.multivariate_normal(means, covs, no_periods).T

    data1=m[0]
    data2=m[1]
    
    empirical_corr=np.corrcoef(data1, data2)[0][1]
    
    
    
    return empirical_corr





## path to difference for one thing no correlation
months_in_year=12
annual_vol=0.270
monthly_vol=annual_vol/(months_in_year**.5)
annual_SR=0.05
annual_SR2=0.05
diffSR=annual_SR - annual_SR2
annual_return=annual_vol*annual_SR
annual_return2=annual_vol*annual_SR2
monthly_mean=annual_return/months_in_year
monthly_mean2=annual_return2/months_in_year

## Make sure these match!
no_periods=36

monte_carlos=1000000
corr=0.75

corrdist=[twocorrelatedseries(no_periods, monthly_mean, monthly_mean2, monthly_vol, corr=corr) for ii in range(monte_carlos)]

print np.percentile(corrdist, 75)
print np.percentile(corrdist, 50)
print np.mean(corrdist)

def linehist(x, color="blue", linestyle="-", bins=10, linewidth=1):
    y,binEdges =np.histogram(x, bins=bins)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    plot(bincenters,y,'-', color=color, linestyle=linestyle, linewidth=linewidth) 

linehist(corrdist, bins=50, linewidth=2)


frame=plt.gca()

frame.get_yaxis().set_visible(False)
frame.set_xlim([0.5, 0.95])
frame.set_xticks([0.6, 0.7, 0.8, 0.9])
frame.set_ylim([0,100000])



frame.annotate("Average correlation", xy=(0.745, 85000),xytext=(0.6, 80000.0), arrowprops=dict(facecolor='black', shrink=0.05), size=18)
frame.annotate("75% confidence\n correlation", xy=(0.8, 83000),xytext=(0.84, 80000.0), arrowprops=dict(facecolor='black', shrink=0.05), size=18)
frame.annotate("Breakeven \n with costs", xy=(0.92, 500),xytext=(0.855, 40000.0), arrowprops=dict(facecolor='black', shrink=0.05), size=18)



plt.axvline(0.75, linestyle="--")
plt.axvline(0.8, linestyle="--")
plt.axvline(0.92, linestyle="--")

    
#xlabel("Difference in annual % returns between managers")

rcParams.update({'font.size': 18})



def file_process(filename):
    fig = plt.gcf()
    fig.set_size_inches(18.5,10.5)
    fig.savefig("/home/rob/%s.png" % filename,dpi=300)
    fig.savefig("/home/rob/%sLOWRES.png" % filename,dpi=50)
    
    Image.open("/home/rob/%s.png" % filename).convert('L').save("/home/rob/%s.jpg" % filename)
    Image.open("/home/rob/%sLOWRES.png" % filename).convert('L').save("/home/rob/%sLOWRES.jpg" % filename)


file_process("divbenefits")

show()
