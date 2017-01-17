import numpy as np

def variance_f(sigma, weights):
    return 1.0/((np.matrix(weights)*sigma*np.matrix(weights).transpose())[0,0]**.5)

def make_corr(dimension, offdiag=0.0):

    corr=np.array([[offdiag]*dimension]*dimension)
    corr[np.diag_indices(dimension)]=1.0
    return corr

power_law_set=[
(0.55,      -0.98),
(0.5,    -0.92),
(0.4,    -0.75),
(0.3,    -0.58),
(0.2,    -0.36)]

universe_size=100
power=power_law_set[0][1]

start_cap_values=[((rank/100.0)**(power)) for rank in range(universe_size+1)[1:]]
cfactor=0.80
basestd=0.27
basearithmean=0.05
riskfree=0.0

equalweight=False

if equalweight:
    weights=[1.0/universe_size]*universe_size
else:
    sumx=sum(start_cap_values)
    weights=[x/sumx for x in start_cap_values]

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

#print calc_stats(weights, basestd, basearithmean, cfactor=cfactor)

### Use TSX weights
import pandas as pd
tsx_weights=pd.read_csv("/home/rob/workspace/systematictradingexamples/plots_for_perhaps/XIU_holdings.csv")
tsx_weights=tsx_weights.iloc[:60,]
universe_size=60



corrmatrix=np.zeros((universe_size, universe_size))

for i in range(universe_size):
    for j in range(universe_size):
        if i==j:
            corr=1.0
        elif tsx_weights.irow(i).Sector==tsx_weights.irow(j).Sector:
            corr=0.85
        else:
            corr=0.75
        corrmatrix[i][j]=corr

## top down weights
sectors=list(tsx_weights.Sector.unique())
weight_each_sector=1.0/len(sectors)

sector_counts=dict([(sector, sum(tsx_weights.Sector==sector)) for sector in sectors])

topdown_weights=[]
for (tidx, sector) in enumerate(tsx_weights.Sector):
    topdown_weights.append(weight_each_sector * (1.0/sector_counts[sector]))
    
##cap weights
cap_weights=list(tsx_weights.Weight.values)
cap_weights=cap_weights/sum(cap_weights)




equal_weights=[1.0/universe_size]*universe_size

# (gmm, new_std, gsr, cost_gmm, cost_gsr)
print calc_stats(cap_weights, basestd, basearithmean, corrmatrix)
print calc_stats(equal_weights, basestd, basearithmean, corrmatrix)
print calc_stats(topdown_weights, basestd, basearithmean, corrmatrix)



## randomness
## only need to this one once
random_topdown_weights =[]
done_this_sector=[]
for (tidx, sector) in enumerate(tsx_weights.Sector):
    if sector in done_this_sector:
        random_topdown_weights.append(0.0)
    else:
        random_topdown_weights.append(weight_each_sector)
        done_this_sector.append(sector)
        
print calc_stats(random_topdown_weights, basestd, basearithmean, corrmatrix)
    
psize=10
        
import random
### multiple times
all_gmeans=[]
all_sr=[]

for notused in range(10000):
    things=[int(np.ceil(random.uniform(0,61))) for notused2 in range(psize*2)]
    things=list(set(things)) ## make unique
    things=things[:psize] ##only need 10 
    random_weights=[]
    for (tidx, sector) in enumerate(tsx_weights.Sector):
        if tidx in things:
            random_weights.append(1.0/psize)
        else:
            random_weights.append(0.0)
    stats = calc_stats(random_weights, basestd, basearithmean, corrmatrix)
    all_gmeans.append(stats[0])
    all_sr.append(stats[2])
    
np.mean(all_gmeans)
np.mean(all_sr)
