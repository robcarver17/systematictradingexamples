from scipy.optimize import minimize
from copy import copy
import random

import pandas as pd
import numpy as np
from datetime import datetime as dt

def create_dull_pd_matrix(dullvalue=0.0, dullname="A", startdate=pd.datetime(1970,1,1).date(), enddate=dt.now().date(), index=None):
    """
    create a single valued pd matrix
    """
    if index is None:
        index=pd.date_range(startdate, enddate)
    
    dullvalue=np.array([dullvalue]*len(index))
    
    ans=pd.DataFrame(dullvalue, index, columns=[dullname])
    
    return ans

def addem(weights):
    ## Used for constraints
    return 1.0 - sum(weights)

def variance(weights, sigma):
    ## returns the variance (NOT standard deviation) given weights and sigma
    return (np.matrix(weights)*sigma*np.matrix(weights).transpose())[0,0]


def neg_SR(weights, sigma, mus):
    ## Returns minus the Sharpe Ratio (as we're minimising)

    """    
    estreturn=250.0*((np.matrix(x)*mus)[0,0])
    variance=(variance(x,sigma)**.5)*16.0
    """
    estreturn=(np.matrix(weights)*mus)[0,0]
    std_dev=(variance(weights,sigma)**.5)

    
    return -estreturn/std_dev

def sigma_from_corr(std, corr):
    sigma=std*corr*std

    return sigma

def basic_opt(std,corr,mus):
    number_assets=mus.shape[0]
    sigma=sigma_from_corr(std, corr)
    start_weights=[1.0/number_assets]*number_assets
    
    ## Constraints - positive weights, adding to 1.0
    bounds=[(0.0,1.0)]*number_assets
    cdict=[{'type':'eq', 'fun':addem}]

    return minimize(neg_SR_riskfree, start_weights, (sigma, mus), method='SLSQP', bounds=bounds, constraints=cdict, tol=0.00001)

def neg_SR_riskfree(weights, sigma, mus, riskfree=0.005):
    ## Returns minus the Sharpe Ratio (as we're minimising)

    """    
    estreturn=250.0*((np.matrix(x)*mus)[0,0])
    variance=(variance(x,sigma)**.5)*16.0
    """
    estreturn=(np.matrix(weights)*mus)[0,0] - riskfree
    std_dev=(variance(weights,sigma)**.5)

    
    return -estreturn/std_dev


def equalise_vols(returns, default_vol):
    """
    Normalises returns so they have the in sample vol of defaul_vol (annualised)
    Assumes daily returns
    """
    
    factors=(default_vol/16.0)/returns.std(axis=0)
    facmat=create_dull_pd_matrix(dullvalue=factors, dullname=returns.columns, index=returns.index)
    norm_returns=returns*facmat
    norm_returns.columns=returns.columns

    return norm_returns


def offdiag_matrix(offvalue, nlength):
    identity=np.diag([1.0]*nlength)
    for x in range(nlength):
        for y in range(nlength):
            if x!=y:
                identity[x][y]=offvalue
    return identity

def get_avg_corr(sigma):
    new_sigma=copy(sigma)
    np.fill_diagonal(new_sigma,np.nan)
    return np.nanmean(new_sigma)



def read_ts_csv(fname, dindex="Date"):
    data=pd.read_csv(fname)
    dateindex=[dt.strptime(dx, "%d/%m/%y") for dx in list(data[dindex])]
    data.index=dateindex
    del(data[dindex])
    
    return data

def markosolver(returns, equalisemeans=False, equalisevols=True, default_vol=0.2, default_SR=1.0):
    """
    Returns the optimal portfolio for the dataframe returns
    
    If equalisemeans=True then assumes all assets have same return if False uses the asset means    
    
    If equalisevols=True then normalises returns to have same standard deviation; the weights returned
       will be 'risk weightings'
       
    Note if usemeans=True and equalisevols=True effectively assumes all assets have same sharpe ratio
    
    """
        
    if equalisevols:
        use_returns=equalise_vols(returns, default_vol)
    else:
        use_returns=returns
    

    ## Sigma matrix 
    sigma=use_returns.cov().values

    ## Expected mean returns    
    est_mus=[use_returns[asset_name].mean() for asset_name in use_returns.columns]
    missingvals=[np.isnan(x) for x in est_mus]
    
        
    if equalisemeans:
        ## Don't use the data - Set to the average Sharpe Ratio
        mus=[default_vol*default_SR]*returns.shape[1]

    else:
        mus=est_mus
    
    mus=np.array(mus, ndmin=2).transpose()

    
    ## Starting weights
    number_assets=use_returns.shape[1]
    start_weights=[1.0/number_assets]*number_assets
    
    ## Constraints - positive weights, adding to 1.0
    bounds=[(0.0,1.0)]*number_assets
    cdict=[{'type':'eq', 'fun':addem}]
    
    ans=minimize(neg_SR, start_weights, (sigma, mus), method='SLSQP', bounds=bounds, constraints=cdict, tol=0.00001)
    wts=ans['x']
        
    return wts

def bootstrap_portfolio(returns_to_bs, monte_carlo=500, monte_length=25, equalisemeans=False, equalisevols=True, default_vol=0.2, default_SR=1.0):
    
    """
    Given dataframe of returns; returns_to_bs, performs a bootstrap optimisation
    
    We run monte_carlo numbers of bootstraps
    Each one contains monte_length days drawn randomly, with replacement 
    (so *not* block bootstrapping)
    
    The other arguments are passed to the optimisation function markosolver
    
    Note - doesn't deal gracefully with missing data. Will end up downweighting stuff depending on how
      much data is missing in each boostrap. You'll need to think about how to solve this problem. 
    
    """
    
    
    weightlist=[]
    for unused_index in range(monte_carlo):
        bs_idx=[int(random.uniform(0,1)*len(returns_to_bs)) for i in range(monte_length)]
        
        returns=returns_to_bs.iloc[bs_idx,:] 
        weight=markosolver(returns, equalisemeans=equalisemeans, equalisevols=equalisevols, default_vol=default_vol, default_SR=default_SR)
        weightlist.append(weight)
        
    ### We can take an average here; only because our weights always add up to 1. If that isn't true
    ###    then you will need to some kind of renormalisation
     
    theweights_mean=list(np.mean(weightlist, axis=0))
    return theweights_mean


rawdata=read_ts_csv("/home/rob/workspace/systematictradingexamples/plots_for_perhaps/sectorreturns.csv")
rawdata=rawdata/100.0

bootstrap_portfolio(rawdata,  equalisemeans=True, equalisevols=True, default_vol=0.2, default_SR=1.0)