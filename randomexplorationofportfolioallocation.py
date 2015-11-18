"""
Explore portfolio optimisation
"""
import random
import matplotlib.pyplot as plt
from commonrandom import threeassetportfolio
from optimisation import opt_shrinkage, bootstrap_portfolio, handcrafted
import numpy as np
from common import DAYS_IN_YEAR, ROOT_DAYS_IN_YEAR
import pandas as pd

## Generate random data with various characteristics
## each tuple is sharpes, then correlations
portfolio=([ .5, .5, .5],[0.8, 0.0, 0.0])

years_to_test=[1, 5, 10, 20]

monte_length=200
test_period_years=1
## arbitrary
methods=["HC", "BS", "S00", "S03", "S30","S33","S06" ,"S60","S36","S63", "S66","S01", "S10",   "S31",   "S61","S13", "S16", "S11"]

## used for plot order
methods_ms=[ "S00", "S03", "S06", "S01", "S30", "S33", "S36",  "S31", "S60", "S63", "S66", "S61", "S10", "S13", "S16", "S11", "HC", "BS"]

methods_by_cs=[ "S00", "S30", "S60", "S10", "S03", "S33", "S63", "S13", "S06", "S36" ,"S66", "S16", "S01", "S31", "S61", "S11", "HC", "BS"]

benchmarket="S00"
sfactor_lookup=dict([("1",1.0), ("0", 0.0), ("3", 0.333), ("6", 0.66) ])

## For the shorter periods we just use earlier examples of data we've already tested
max_length_years=np.max(years_to_test)+1
plength=max_length_years*DAYS_IN_YEAR+100
test_period_days=test_period_years*DAYS_IN_YEAR

SRlist, clist=portfolio
all_returns=[threeassetportfolio(plength=plength, SRlist=SRlist, clist=clist) for notused in range(monte_length)]


for data_length in years_to_test:
    print "Testing data length %d" % data_length
    in_sample_start=0
    in_sample_end=data_length*DAYS_IN_YEAR
    out_sample_start=in_sample_end+1
    out_sample_end=out_sample_start+test_period_days
    
    
    answers_this_port_SR=[np.nan]
    answers_this_port_degrade=[np.nan]

    returns_to_use_insample=[one_return_series.iloc[in_sample_start:in_sample_end] for one_return_series in all_returns]
    returns_to_use_outsample=[one_return_series.iloc[out_sample_start:out_sample_end] for one_return_series in all_returns]
    
    len_returns_to_use_os=[one_returns.shape[0] for one_returns in returns_to_use_outsample]
    len_returns_to_use_is=[one_returns.shape[0] for one_returns in returns_to_use_insample]

    
    """
    Run each method over
    """

    
    for method_name in methods:
        
        if method_name=="HC":
            wts=[handcrafted(one_returns) for one_returns in returns_to_use_insample]
        elif method_name=="BS":
            wts=[bootstrap_portfolio(one_returns, monte_carlo=50, monte_length=int(data_length*DAYS_IN_YEAR*.1), equalisemeans=False, equalisevols=False)
                 for one_returns in returns_to_use_insample]
        elif method_name[0]=="S":
            ## shrinkage
            shrinkage_factors=(sfactor_lookup[method_name[1]], sfactor_lookup[method_name[2]])
            wts=[opt_shrinkage(one_returns, shrinkage_factors, equalisevols=False) for one_returns in returns_to_use_insample]
        else:
            raise Exception("Not recognised")
        
        ## See how these weights did in and out of sample
        wts_mat_os=[np.array([list(weights)]*length_needed, ndmin=2) for (weights, length_needed) in zip( wts, len_returns_to_use_os)]
        wts_mat_is=[np.array([list(weights)]*length_needed, ndmin=2) for (weights, length_needed) in zip( wts, len_returns_to_use_is)]
        perf_os=[weight_mat*returns.values for (weight_mat, returns) in zip(wts_mat_os, returns_to_use_outsample)]
        perf_is=[weight_mat*returns.values for (weight_mat, returns) in zip(wts_mat_is, returns_to_use_insample)]
        
        perf_os=[perf.sum(axis=1) for perf in perf_os]
        perf_is=[perf.sum(axis=1) for perf in perf_is]
        
        ann_return_os=[ROOT_DAYS_IN_YEAR*perf.mean() for perf in perf_os]
        sharpe_ann_return_os=np.mean(ann_return_os)/np.std(ann_return_os)
        
        sharpe_os=[ROOT_DAYS_IN_YEAR*perf.mean()/perf.std() for perf in perf_os]
        sharpe_is=[ROOT_DAYS_IN_YEAR*perf.mean()/perf.std() for perf in perf_is]
        
        degradation=[s_os - s_is for (s_os, s_is) in zip(sharpe_os, sharpe_is)]
        avg_degradation=np.mean(degradation)
        
        ## Stack up the results
        
        answers_this_port_SR.append(sharpe_ann_return_os)
        answers_this_port_degrade.append(avg_degradation)
    
    
    results=pd.DataFrame(np.array([answers_this_port_SR, answers_this_port_degrade]).transpose(), columns=["SR", "Degradation"], index=[""]+methods)

    for (methods_idx, mtitle) in zip([methods_ms, methods_by_cs], ["mean shrinkage", "corr shrinkage"]):

        results=results.loc[[""]+methods_idx]
        
        ax=results.plot(use_index=False)
        plt.xticks(range(len(methods_idx)+1))
        plt.title("SR out of sample; Degradation out vs in sample; fit over %d years, order %s" % (data_length, mtitle))
        ax.set_xticklabels([""]+methods_idx)
        plt.show()
        
