

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

from optimisation import *

def creturn(weights, mus):
    return (np.matrix(weights)*mus)[0,0]

def csd(weights, sigma):
    return (variance(weights,sigma)**.5)

def cgm(weights, mus, std_dev):
    estreturn=creturn(weights, mus)

    return estreturn - 0.5*(std_dev**2)


def geo_SR(weights, sigma, mus):
    ## Returns minus the Sharpe Ratio (as we're minimising)

    """    
    estreturn=250.0*((np.matrix(x)*mus)[0,0])
    variance=(variance(x,sigma)**.5)*16.0
    """
    std_dev=csd(weights, sigma)
    geomean = cgm(weights,  mus, std_dev)
    
    return -geomean/std_dev

def arith_SR(weights, sigma, mus):
    ## Returns minus the Sharpe Ratio (as we're minimising)

    """    
    estreturn=250.0*((np.matrix(x)*mus)[0,0])
    variance=(variance(x,sigma)**.5)*16.0
    """
    std_dev=csd(weights, sigma)
    amean = creturn(weights,  mus)
    
    return -amean/std_dev

def basic_opt(std,corr,mus):
    number_assets=mus.shape[0]
    sigma=sigma_from_corr(std, corr)
    start_weights=[1.0/number_assets]*number_assets
    
    ## Constraints - positive weights, adding to 1.0
    bounds=[(0.0,1.0)]*number_assets
    cdict=[{'type':'eq', 'fun':addem}]

    return minimize(geo_SR, start_weights, (sigma, mus), method='SLSQP', bounds=bounds, constraints=cdict, tol=0.00001)


mus=np.array([[0.04,0.04, .0625]]).transpose()
std=np.array([[0.08,0.08,.12]]).transpose()
#mus=np.array([[0.0186,0.0827]]).transpose()
#std=np.array([[0.0623,0.198]]).transpose()

c2=0.9
c1=.9
c3=0.9
corr=np.array([[1,c1,c2],[c1,1.0,c3],[c2,c3,1.0]])

print basic_opt(std,corr,mus)



mus=np.array([[0.016, 0.05]]).transpose()
std=np.array([[0.0827,0.198]]).transpose()

c1=.05
corr=np.array([[1,c1],[c1,1.0]])

def wtfunc(bondweight):
    return [bondweight, 1.0 - bondweight]

rets=[]
sds=[]
gms=[]
gsr=[]
asr=[]

bondwts=np.arange(0.0, 1.0, 0.01)

for bondweight in bondwts:
    weights=wtfunc(bondweight)
    sigma=sigma_from_corr(std, corr)
    std_dev=csd(weights, sigma)
    
    sds.append(std_dev)
    rets.append(creturn(weights, mus))
    gms.append(cgm(weights, mus, std_dev))
    
    gsr.append(-geo_SR(weights, sigma, mus))
    asr.append(-arith_SR(weights, sigma, mus))


def file_process(filename):
    fig = plt.gcf()
    fig.set_size_inches(18.5,10.5)
    fig.savefig("/home/rob/%s.png" % filename,dpi=300)
    fig.savefig("/home/rob/%sLOWRES.png" % filename,dpi=50)
    
    Image.open("/home/rob/%s.png" % filename).convert('L').save("/home/rob/%s.jpg" % filename)
    Image.open("/home/rob/%sLOWRES.png" % filename).convert('L').save("/home/rob/%sLOWRES.jpg" % filename)




plot(bondwts, gms)
frame=plt.gca()

#frame.set_xticks([0.6, 0.7, 0.8, 0.9])
frame.set_ylim([0.01, 0.04])
frame.set_yticks([0.01, 0.02, 0.03])
rcParams.update({'font.size': 18})
file_process("wts_rets_new")

plt.show()

plot(bondwts, sds)
frame=plt.gca()

frame.set_ylim([0.05, 0.2])
frame.set_yticks([0.05, 0.1, 0.15, 0.2])

rcParams.update({'font.size': 18})
file_process("wts_sds")

plt.show()

plot(bondwts, gsr)
frame=plt.gca()
frame.set_ylim([0.2, 0.5])
frame.set_yticks([0.2, 0.3, 0.4, 0.5])

rcParams.update({'font.size': 18})
file_process("wts_sr")

plt.show()

print basic_opt(std,corr,mus)