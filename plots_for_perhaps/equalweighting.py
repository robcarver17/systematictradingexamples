import numpy as np
import pandas as pd

from matplotlib.pyplot import show

"""
prove that equal weighting is better, and by how much
"""

def boring_corr_matrix(size, offdiag=0.99, diag=1.0):
    size_index=range(size)
    def _od(offdag, i, j):
        if i==j:
            return diag
        else:
            return offdiag
    m= [[_od(offdiag, i,j) for i in size_index] for j in size_index]
    m=np.array(m)
    return m

def sigma_from_corr_and_std(stdev, corrmatrix):
    
    sigma=stdev*corrmatrix*stdev

    return sigma




def correlatedseries(no_periods, means, covs):

    m = np.random.multivariate_normal(means, covs, no_periods)

    return m


def update_portfolio_values(prior_values,  return_ts, new_date_index):
    """
    Update portfolio values from one period to the next
    
    These follow returns unless we're rebalancing
    
    Zero weights stay at zero
    """
    relevant_returns=list(return_ts.iloc[new_date_index,:])
    
    new_values=[old_value*(1+this_return) for (old_value, this_return) in zip(prior_values, relevant_returns)]
    
    return new_values

    
    
def return_correct_values(current_values, correct_weights):
    """
    do a rebalance, assuming weights add up to 1
    """
    total_current_value=sum(current_values)
    correct_values=[total_current_value*weight for weight in correct_weights]
    
    return correct_values

def index_weights_period(market_cap_values,
                              new_date_index, index_rank_in, index_rank_out, rebalance_buffer_not_used,
                              index_size, prior_values=None):
    """
    Generate a set of index weights given:
      - current market cap valuations
      - indexing rules
    """
    if prior_values is None:
        prior_values=[np.nan]*market_cap_values.shape[1]

    current_cap_values=market_cap_values.iloc[new_date_index,:]
    new_rankings=generate_rankings(current_cap_values)
    
    ## returns 1 or 0 if in index or not
    in_index_list=[index_weights_value(current_portfolio_value, correct_ranking, index_size, index_rank_in, index_rank_out)
                   for (current_portfolio_value, correct_ranking) in zip(prior_values, new_rankings)]

    ## Market cap weightings
    new_weights=[in_index_flag*correct_index_weight for (in_index_flag, correct_index_weight) in 
                zip(in_index_list, current_cap_values)]
    
    ## normalise
    total_new_weights=sum(new_weights)
    new_weights=[x/total_new_weights for x in new_weights]
    
    return new_weights

def index_weights_value(current_portfolio_value, correct_ranking, 
                        index_size, index_rank_in, index_rank_out):
    """
    Return a 1 or a 0 depending on if something should be ranked or not
    """
    
    if np.isnan(current_portfolio_value):
        if correct_ranking<=index_size:
            return 1
        else:
            return 0
         
    if current_portfolio_value>0:
        ## in index
        if correct_ranking<=index_rank_out:
            return 1
        else:
            return 0
    else:
        ## not currently in index
        if correct_ranking<=index_rank_in:
            return 1
        else:
            return 0


def generate_rankings(current_values):
    """
    Generate rankings from current valuations for a given date
    """
    
    order = current_values.argsort()
    ranks = order.argsort()
    ranks=list(ranks)
    ranks=[len(ranks)-x for x in ranks]
    
    return ranks

def buffer_weights(ideal_weights, prior_values, rebalance_buffer):
    """
    Buffer weights to reduce trading costs
    """

    implied_portfolio_value=return_correct_values(prior_values, ideal_weights)
    total_portfolio_value=sum(prior_values)
    
    required_trades=[(wanted - have)/total_portfolio_value for (wanted, have) in zip(implied_portfolio_value, prior_values)]
    
    large_trades=[abs(x)>rebalance_buffer for x in required_trades]
    
    if any(large_trades):
        return ideal_weights
    else:
        totalx=sum(prior_values)
        
        if np.isnan(totalx):
            return ideal_weights
        effective_prior_weights=[x/totalx for x in prior_values]
        
        return effective_prior_weights
    

def equal_weights_period(market_cap_values,
                              new_date_index, index_rank_in, index_rank_out, rebalance_buffer,
                              index_size, prior_values=None):
    """
    Generate a set of equal weights given:
      - current market cap valuations (only used for indexing)
      - indexing rules
      - prior values
      - rebalance buffer
    """
    
    equal_weight=1.0/index_size
    
    if prior_values is None:
        ## return equal weights
        prior_values=[np.nan for notused in range(universe_size)]

    current_cap_values=market_cap_values.iloc[new_date_index,:]
    new_rankings=generate_rankings(current_cap_values)
    
    ## returns 1 or 0 if in index or not
    in_index_list=[index_weights_value(current_portfolio_value, correct_ranking, index_size, index_rank_in, index_rank_out)
                   for (current_portfolio_value, correct_ranking) in zip(prior_values, new_rankings)]

    ## Ideal weightings
    ideal_weights=[in_index_flag*equal_weight for in_index_flag in 
                in_index_list]

    ## buffer
    new_weights=buffer_weights(ideal_weights, prior_values, rebalance_buffer)
    
    return new_weights


def increment_period_index(market_cap_values, index_rank_in, index_rank_out, rebalance_buffer,
                          index_size, universe_size, return_ts, 
                          new_date_index, rebalancing_freq,
                          rebalancing_func=index_weights_period,
                          prior_values=None):
    
    """
    Increment each period
    
    We update the 'portfolio value' (cash value of each thing in our portfolio)
    
    We return this as a tuple
    
    The first element of the tuple is the value from natural growth in the portfolio
    The second optional element is the value after any rebalancing has taken place
    
    """
    
    if new_date_index==0:
        correct_weights=rebalancing_func(market_cap_values,
                              new_date_index, index_rank_in, index_rank_out, rebalance_buffer,
                              index_size, prior_values=None)
        
        new_portfolio_value=correct_weights
        ## no rebalancing
        
        return (None, new_portfolio_value)
    
    ## portfolio values will 'drift' unless we rebalance
    new_portfolio_value=update_portfolio_values(prior_values,  return_ts, new_date_index)
    
    ## Check for rebalancing
    cyclepoint=new_date_index % rebalancing_freq
    if cyclepoint==0:
        ## rebalancing
        
        new_weights=rebalancing_func(market_cap_values,
                          new_date_index, index_rank_in, index_rank_out, rebalance_buffer,
                          index_size, prior_values)
    
        rebalanced_values=return_correct_values(new_portfolio_value, new_weights)
        
        return (new_portfolio_value, rebalanced_values)
        
    else:
        return (None, new_portfolio_value)
    


    
    
def generate_values(market_cap_values, index_rank_in, index_rank_out, rebalance_buffer,
                          index_size, universe_size, return_ts,
                          rebalancing_freq, rebalancing_func):
    
    stack_values=[]
    
    new_date_index=0
    
    prior_values=None
    
    while new_date_index<market_cap_values.shape[0]:
        new_values=increment_period_index(market_cap_values, index_rank_in, index_rank_out,rebalance_buffer,
                                  index_size, universe_size, return_ts,
                                  new_date_index, rebalancing_freq,
                                  rebalancing_func,
                                  prior_values)
        
        stack_values.append(new_values)
        new_date_index=new_date_index+1
        prior_values=new_values[1]
        
    return stack_values


def turnover_from_value_item(value_item):
    """
    Returns turnover as a proportion of portfolio size
    """

    if value_item[0] is None:
        return 0.0
    
    (new_portfolio, old_portfolio) = value_item
    
    total_value=sum(old_portfolio)
    
    trades=[abs(new-old)/total_value for (new,old) in zip(new_portfolio, old_portfolio)]

    return sum(trades)

def avg_turnover(value_history):
    """
    Returns the average turnover, per year
    """
    turnover_list=[turnover_from_value_item(value_item) for value_item in value_history]
    
    avg_daily_turnover=np.mean(turnover_list)
    
    return avg_daily_turnover*250


def portfolio_stats(account_values):
    """
    returns some portfolio stats
    """
    ## convert to % returns
    
    perc_returns=(account_values - account_values.shift(1)) / account_values.shift(1)
    
    ann_avg=perc_returns.mean().values[0]*250
    ann_std=perc_returns.std().values[0]*16
    ann_SR=ann_avg/ann_std
    
    return (ann_avg, ann_std, ann_SR)

def get_stats(market_cap_values, index_rank_in, index_rank_out, rebalance_buffer,
                          index_size, universe_size, return_ts, rebalancing_func,
                          rebalancing_freq):
    """
    Generate some statistics for a given data set and rebalancing function
    """

    value_history=generate_values(market_cap_values, index_rank_in, index_rank_out, rebalance_buffer,
                              index_size, universe_size, return_ts,
                              rebalancing_freq, rebalancing_func)
    
    ## generate the values from the list of tuples
    ## for now we ignore costs 
    value_data=[value_item[1] for value_item in value_history]
    value_data_ts=pd.DataFrame(np.array(value_data), market_cap_values.index)
    
    gross_account_value=value_data_ts.sum(axis=1).to_frame()
    
    ## We can work out returns, stdev, sharpe directly from the sum of values
    turnover=avg_turnover(value_history)
    
    stats=portfolio_stats(gross_account_value)
    
    return (stats, turnover)


def get_random_data(start_cap_values, datalength, means, covs):
    ### valuations evolve over time
    start_cap_values_ts=pd.DataFrame(np.array(start_cap_values, ndmin=2), index=[pd.datetime(2000,1,1)])
    return_ts=pd.DataFrame(correlatedseries(datalength, means, covs), 
                           index=pd.date_range(pd.datetime(2000,1,1), periods=datalength))
    
    return_cumulator=return_ts+1.0
    
    market_cap_values=pd.concat([start_cap_values_ts, return_cumulator]).cumprod()
    
    return_ts=pd.concat([pd.DataFrame(np.array([0.0]*universe_size, ndmin=2), index=[pd.datetime(2000,1,1)]),
                         return_ts])
    
    return (market_cap_values, return_ts)

##

monte_count=100

### Starting weights according to a power law
### find the exponent factor

power_law_set=[
(0.55,      -0.98),
(0.5,    -0.92),
(0.4,    -0.75),
(0.3,    -0.58),
(0.2,    -0.36)]

index_size=10
universe_size=10

datalength=2500

rebalancing_freq=62 ## 250 is annual, 125 is 6 monthly, 62 is quarterly, 21 is monthly

## index rules
## these are similar to the FTSE (90th, and 111th)
index_rank_in=int(np.ceil(index_size*0.9))
index_rank_out=int(np.floor(index_size*1.1))

## average returns and vol
avg_ann_return=0.05
avg_ann_stdev=0.25

daily_return=avg_ann_return/250
daily_stdev=avg_ann_stdev/16

means = [daily_return]*universe_size
stdev=np.array([daily_stdev]*universe_size, ndmin=2).transpose()

list_of_buffers=[0.0, 0.01, 0.05, 0.1, 0.2]
correlation_list=[0.6, 0.7, 0.8]

giant_results_stack=[]

## loop through correlations
for avg_correlation in correlation_list:

    corrmatrix=boring_corr_matrix(universe_size, avg_correlation)
    covs=sigma_from_corr_and_std(stdev, corrmatrix)
    
    all_results_this_corr=[]
    
    ## loop through powers
    for power_tuple in power_law_set:
        power=power_tuple[1]
        print("correlation %f" % avg_correlation)

        print("power %f" % power)
    
        ## values don't have to be normalised in any way
        start_cap_values=[((rank/100.0)**(power)) for rank in range(universe_size+1)[1:]]
        
        ## theoretical advantage
        total_cap=sum(start_cap_values)
        wts=[cv/total_cap for cv in start_cap_values]
        theoretical_SR_cap=0.25/((np.matrix(wts)*corrmatrix*np.matrix(wts).transpose())[0,0]**.5)
        
        equalwts=[1.0/universe_size for notused in range(universe_size)]
        theoretical_SR_eq=0.25/((np.matrix(equalwts)*corrmatrix*np.matrix(equalwts).transpose())[0,0]**.5)

        
        print "Theoretical SR: Cap %.4f Equal %.4f" % (theoretical_SR_cap, theoretical_SR_eq)
        
        all_results_this_power=[]
        
        for monteruns in range(monte_count):
            print("monte %d" % monteruns)
            (market_cap_values, return_ts)=get_random_data(start_cap_values, datalength, means, covs)
            
            ans_cap_weighted=get_stats(market_cap_values, index_rank_in, index_rank_out, 0.0,
                                      index_size, universe_size, return_ts, index_weights_period,
                                      rebalancing_freq)
        
            ans_equal_weights=[]
            for rebalance_buffer in list_of_buffers:    
                ans_equal_weights.append(get_stats(market_cap_values, index_rank_in, index_rank_out, rebalance_buffer,
                                          index_size, universe_size, return_ts, equal_weights_period,
                                          rebalancing_freq))
        
            all_ans=(ans_cap_weighted, ans_equal_weights)
            
            all_results_this_power.append(all_ans)
        
        all_results_this_corr.append(all_results_this_power)
        
    giant_results_stack.append(all_results_this_corr)

import pickle
f=open("/home/rob/temp/equalweighting.pck", "wb")
pickle.dump(giant_results_stack, f)
f.close()

## use second buffer for now
## loop through correlations
for (corridx, avg_correlation) in enumerate(correlation_list):

    all_results_this_corr=giant_results_stack[corridx]
    
    ## loop through powers
    for (poweridx, power_tuple) in enumerate(power_law_set):
        power=power_tuple[1]
    
        
        all_results_this_power=all_results_this_corr[poweridx]
        
        all_cap=[]
        all_equal=[]
        for midx in range(monte_count):
            
            (ans_cap_weighted, ans_equal_weights)=all_results_this_power[midx]
            
            ans_equal_weights_use=ans_equal_weights[1]
            
            all_cap.append(ans_cap_weighted)
            all_equal.append(ans_equal_weights_use)
        
        def _avg_in_tuple_list(tuple_list, tidx1, tidx2=None):   
            
            def _extract(tuple_item, tidx1, tidx2=None):
                if tidx2 is None:
                    return tuple_item[tidx1]
                else:
                    return tuple_item[tidx1][tidx2]
                
            tlist=[_extract(tuple_item, tidx1, tidx2) for tuple_item in tuple_list]
            
            return np.mean(tlist)

        print("correlation %f" % avg_correlation)

        print("power %f" % power)
        
        print("SR Cap %.4f  Equal %.4f" % (_avg_in_tuple_list(all_cap, 0, 2), _avg_in_tuple_list(all_equal, 0, 2)))
        print("Return Cap %.4f  Equal %.4f" % (_avg_in_tuple_list(all_cap, 0, 0), _avg_in_tuple_list(all_equal, 0, 0)))
        print("Vol Cap %.4f  Equal %.4f" % (_avg_in_tuple_list(all_cap, 0, 1), _avg_in_tuple_list(all_equal, 0, 1)))
        print("Turnover Cap %.4f  Equal %.4f" % (_avg_in_tuple_list(all_cap, 1), _avg_in_tuple_list(all_equal, 1)))