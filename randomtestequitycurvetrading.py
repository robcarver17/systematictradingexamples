"""
Test equity curve trading
"""

import numpy as np
from commonrandom import autocorr_skewed_returns, adj_moments_for_rho
from matplotlib.pyplot import plot, title, show, xticks, legend
from common import account_curve, arbitrary_timeseries
import pandas as pd

NO_OVERLAY=1000



## Use period_length=1, peiod_length=5 for weeks, 21 for months
## Choose scenario type - "VaryingSharpeRatio", "VaryingSkew", "VaryingAuto"
scenario_type="VaryingAutoMore"
period_length=1

annualised_costs_SR=0.03

## Higher number improves accuracy of results
monte_runs=100
curve_length_years=20


def divxx(x):
    if x[1]==0.0:
        return np.nan
    
    return x[0]/-x[1]
    
def isbelow(cum_x, mav_x, idx):
    ## returns 1 if cum_x>mav_x at idx, 0 otherwise
    if cum_x[idx]>=mav_x[idx]:
        return 1.0
    return 0.0

def apply_overlay(x, N_length, period_stdev, costs_SR=0):
    """
    apply an equity curve filter overlay
    
    x is a pd time series of returns
    
    N_length is the mav to apply
    
    Returns a new x with 'flat spots'
    
    """
    if N_length==NO_OVERLAY:
        return x

    cum_x=x.cumsum()
    mav_x=pd.rolling_mean(cum_x, N_length)
    filter_x=pd.TimeSeries([isbelow(cum_x, mav_x, idx) for idx in range(len(x))], x.index)
    
    ## can only apply with a lag (!)
    filtered_x=x*filter_x.shift(1)
    
    if costs_SR>0:
        ## apply costs
        ## first work out the turnover
        ## note everything is adjusted for the period we are in

        turnover=filter_x.diff().abs().mean()/2.0
        
        return_degrade=costs_SR*turnover
        filtered_x = filtered_x - return_degrade
    
    return filtered_x

def plot_array(data_list, array_title, N_windows, scenario_list):

    data_array=np.array(data_list).transpose()
    
    plot(data_array)
    xticks(range(data_array.shape[0]), N_windows)
    legend([x[0] for x in scenario_list])
    
    title(array_title)
    show()




if scenario_type=="VaryingSharpeRatio":
    ##fixed characteristics
    
    skew_list=[0.0]
    autocorr_list=[0.0]
    
    ## varying characteristics
    ann_sharpe_list=[-1.0, 0.0,  1.0,  2.0]
    
elif scenario_type=="VaryingSkew":
    autocorr_list=[0.0]
    ann_sharpe_list=[1.0]
    
    ## varying characteristics
    skew_list=[-2.0, 0.0, 1.0]

    
elif scenario_type=="VaryingAuto":
    skew_list=[0.0]
    ann_sharpe_list=[1.0]
    
    ## varying characteristics
    autocorr_list=[-0.3, 0.0, 0.3]

elif scenario_type=="VaryingAutoMore":
    skew_list=[0.0]
    ann_sharpe_list=[1.0]
    
    ## varying characteristics
    autocorr_list=[-0.3,  0.0, 0.1, 0.2, 0.3]


    
else:
    raise Exception("%s?" % scenario_type)

## Calculations...
## Fixed
ann_stdev=0.20
min_N_size=3
curve_length=curve_length_years*256/period_length

N_windows=[10,25,40, 64, 128, 256, 512]

root_period_adj=(256.0/period_length)**.5
period_adj=256.0/period_length
period_stdev=ann_stdev/root_period_adj
N_windows=[int(np.ceil(x/period_length)) for x in N_windows]
N_windows=[x for x in N_windows if x>(min_N_size-1)]
period_mean=[ann_stdev*x/period_adj for x in ann_sharpe_list]

costs_SR=annualised_costs_SR/root_period_adj


## Each scenario is a tuple: 'name', mean return, stdev returns, skew, autocorrelation 
if scenario_type=="VaryingSharpeRatio":
    scenario_list=[("SR:%.1f" % ann_sharpe_list[idx], period_mean[idx], period_stdev, skew_list[0], autocorr_list[0]) for idx in range(len(ann_sharpe_list))]

elif scenario_type=="VaryingSkew":
    scenario_list=[("Skew:%.1f" % skew_list[idx], period_mean[0], period_stdev, skew_list[idx], autocorr_list[0]) for idx in range(len(skew_list))]

elif scenario_type=="VaryingAuto" or scenario_type=="VaryingAutoMore":
    scenario_list=[("Rho:%.1f" % autocorr_list[idx], period_mean[0], period_stdev, skew_list[0], autocorr_list[idx]) for idx in range(len(autocorr_list))]

else:
    raise Exception("%s?" % scenario_type)

N_windows=N_windows+[NO_OVERLAY]

"""
These lists are where we stack up the stuff we are plotting
"""
avg_annual_returns=[]
avg_drawdowns=[]
max_drawdowns=[]
avg_avg_ratio=[]
avg_max_ratio=[]

for scenario in scenario_list:
    print "Scenario:"
    print scenario
    (notUsed, want_mean, want_stdev, want_skew, want_rho)=scenario
    (adj_want_mean, adj_want_skew, adj_want_stdev)=adj_moments_for_rho(want_rho, want_mean, want_skew, want_stdev)
    
    avg_annual_returns_this_scenario=[]
    avg_drawdowns_this_scenario=[]
    max_drawdowns_this_scenario=[]
    avg_avg_ratio_this_scenario=[]
    avg_max_ratio_this_scenario=[]
    
    ## Window '1000' is no ovelay at all
    for N_length in N_windows:
        print "N_length: %d" % N_length
        ## Generate a lot of random data
        many_curves_raw=[arbitrary_timeseries(autocorr_skewed_returns(want_rho, adj_want_mean, adj_want_stdev, adj_want_skew, size=curve_length)) for notused in range(monte_runs)]
        
        ## Apply the overlay
        many_curves=[account_curve(apply_overlay(x, N_length, period_stdev, costs_SR)) for x in many_curves_raw]
        
        ## Calc statistics
        annual_returns_these_curves=[np.mean(x)*period_adj for x in many_curves]
        avg_drawdowns_these_curves=[x.avg_drawdown() for x in many_curves]
        max_drawdowns_these_curves=[x.worst_drawdown() for x in many_curves]
        
        ## Add results
        avg_annual_returns_this_scenario.append(np.mean(annual_returns_these_curves))
        avg_drawdowns_this_scenario.append(np.nanmean(avg_drawdowns_these_curves))
        max_drawdowns_this_scenario.append(np.nanmean(max_drawdowns_these_curves))
        avg_avg_ratio_this_scenario.append(np.nanmean([divxx(x) for x in zip(annual_returns_these_curves, avg_drawdowns_these_curves)]))
        avg_max_ratio_this_scenario.append(np.nanmean([divxx(x) for x in zip(annual_returns_these_curves, max_drawdowns_these_curves)]))
        
    avg_annual_returns.append(avg_annual_returns_this_scenario)
    avg_drawdowns.append(avg_drawdowns_this_scenario)
    max_drawdowns.append(max_drawdowns_this_scenario)
    avg_avg_ratio.append(avg_avg_ratio_this_scenario)
    avg_max_ratio.append(avg_max_ratio_this_scenario)


    
plot_array(avg_annual_returns, "Average annual return", N_windows, scenario_list)
plot_array(avg_drawdowns, "Average drawdowns", N_windows, scenario_list)
plot_array(max_drawdowns, "Max drawdowns", N_windows, scenario_list)
plot_array(avg_avg_ratio, "Average return / Average drawdown", N_windows, scenario_list)
plot_array(avg_max_ratio, "Average return / Max drawdown", N_windows, scenario_list)
    