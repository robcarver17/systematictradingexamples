
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
    
    return np.mean(data1) - np.mean(data2)





## path to difference for one thing no correlation
months_in_year=12
annual_vol=0.150
monthly_vol=annual_vol/(months_in_year**.5)
annual_SR=0.66
annual_SR2=0.46
diffSR=annual_SR - annual_SR2
annual_return=annual_vol*annual_SR
annual_return2=annual_vol*annual_SR2
monthly_mean=annual_return/months_in_year
monthly_mean2=annual_return2/months_in_year

## Make sure these match!
no_years=10
no_periods=months_in_year*no_years

monte_carlos=500000
corr=0.85

rollmeandiff=[twocorrelatedseries(no_periods, monthly_mean, monthly_mean2, monthly_vol, corr=corr) for ii in range(monte_carlos)]
rollanndiff=[x*12 for x in rollmeandiff]

rollannSR=[x/annual_vol for x in rollanndiff]


def linehist(x, color="blue", linestyle="-", bins=10, linewidth=1):
    y,binEdges =np.histogram(x, bins=bins)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    plot(bincenters,y,'-', color=color, linestyle=linestyle, linewidth=linewidth) 

linehist(rollanndiff, bins=50, linewidth=2)
frame=plt.gca()

frame.get_yaxis().set_visible(False)
frame.set_xlim([-0.07, 0.13])
frame.set_xticks([-0.05, 0.00, 0.05,.1])
frame.set_ylim([0,50000])


frame.annotate("Expected improvement in returns", xy=(0.03, 38000),xytext=(0.05, 45000.0), arrowprops=dict(facecolor='black', shrink=0.05), size=18)
frame.annotate("Breakeven in costs", xy=(0.01, 28000),xytext=(-0.05, 40000.0), arrowprops=dict(facecolor='black', shrink=0.05), size=18)


plt.axvline(0.01, linestyle="--")
plt.axvline(0.03, linestyle="--")

    
#xlabel("Difference in annual % returns between managers")

rcParams.update({'font.size': 18})



def file_process(filename):
    fig = plt.gcf()
    fig.set_size_inches(18.5,10.5)
    fig.savefig("/home/rob/%s.png" % filename,dpi=300)
    fig.savefig("/home/rob/%sLOWRES.png" % filename,dpi=50)
    
    Image.open("/home/rob/%s.png" % filename).convert('L').save("/home/rob/%s.jpg" % filename)
    Image.open("/home/rob/%sLOWRES.png" % filename).convert('L').save("/home/rob/%sLOWRES.jpg" % filename)


file_process("correlateddifferences")

show()

print(sum([1.0 for x in rollanndiff if x<0.01])/len(rollanndiff))
print(np.std(rollanndiff))