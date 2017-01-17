import numpy as np

def variance_f(sigma, weights):
    return 1.0/((np.matrix(weights)*sigma*np.matrix(weights).transpose())[0,0]**.5)

def make_corr(dimension, offdiag=0.0):

    corr=np.array([[offdiag]*dimension]*dimension)
    corr[np.diag_indices(dimension)]=1.0
    return corr


cfactor=0.80
basestd=0.25
basearithmean=0.05
riskfree=0.005


def calc_stats(weights, basestd, basearithmean, corrmatrix=None, cfactor=1.0):
    if corrmatrix is None:
        universe_size=len(weights)
        corrmatrix=make_corr(universe_size, cfactor)
    div_factor=variance_f(corrmatrix, weights)
    new_std=basestd/div_factor
    variance=new_std**2
    gmm=basearithmean- variance/2.0
    gsr=(gmm - riskfree) / new_std
    
    return (gmm, new_std, gsr)


### Use MSCI world weights
import pandas as pd
tsx_weights=pd.read_csv("/home/rob/workspace/systematictradingexamples/plots_for_perhaps/MSCIWorldweights.csv")

universe_size=len(tsx_weights.index)

corrmatrix=np.zeros((universe_size, universe_size))

for i in range(universe_size):
    for j in range(universe_size):
        if i==j:
            corr=1.0
        elif tsx_weights.irow(i).Region==tsx_weights.irow(j).Region:
            corr=0.85
        else:
            corr=0.60
        corrmatrix[i][j]=corr

## top down weights
regions=list(tsx_weights.Region.unique())
weight_each_region=1.0/len(regions)

region_counts=dict([(region, sum(tsx_weights.Region==region)) for region in regions])

topdown_weights=[]
for (tidx, region) in enumerate(tsx_weights.Region):
    topdown_weights.append(weight_each_region * (1.0/region_counts[region]))
    
##cap weights
cap_weights=list(tsx_weights.weight.values)
cap_weights=cap_weights/sum(cap_weights)
equal_weights=[1.0/universe_size]*universe_size

print calc_stats(cap_weights, basestd, basearithmean, corrmatrix)
print calc_stats(equal_weights, basestd, basearithmean, corrmatrix)
print calc_stats(topdown_weights, basestd, basearithmean, corrmatrix)
