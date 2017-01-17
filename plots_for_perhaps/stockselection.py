import numpy as np

def variance_f(sigma, weights):
    return 1.0/((np.matrix(weights)*sigma*np.matrix(weights).transpose())[0,0]**.5)

def make_corr(dimension, offdiag=0.0):

    corr=np.array([[offdiag]*dimension]*dimension)
    corr[np.diag_indices(dimension)]=1.0
    return corr


def calc_stats(weights, basestd, basearithmean, corrmatrix=None, cfactor=1.0, cost=0.0, riskfree=0.0):
    if corrmatrix is None:
        universe_size=len(weights)
        corrmatrix=make_corr(universe_size, cfactor)
    div_factor=variance_f(corrmatrix, weights)
    new_std=basestd/div_factor
    variance=new_std**2
    gmm=basearithmean- variance/2.0
    gsr=(gmm - riskfree) / new_std
    
    cost_gmm = gmm - cost
    cost_gsr = (cost_gmm - riskfree) / new_std
    
    return (gmm, new_std, gsr, cost_gmm, cost_gsr)


### Use TSX weights
import pandas as pd
sector_count=11
stocks_per_sector=2
universe_size = sector_count * stocks_per_sector

corrmatrix=np.zeros((universe_size, universe_size))

insector=sum([range(sector_count)]*stocks_per_sector,[])

for i in range(universe_size):
    for j in range(universe_size):
        if i==j:
            corr=1.0
        elif insector[i]==insector[j]:
            corr=0.85
        else:
            corr=0.75
        corrmatrix[i][j]=corr

## top down weights
equal_weights=[1.0/universe_size]*universe_size

# (gmm, new_std, gsr, cost_gmm, cost_gsr)
basestd=.27
basearithmean=.05
print calc_stats(equal_weights, basestd, basearithmean, corrmatrix)
