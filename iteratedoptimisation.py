"""
This file shows how to do iterated portfolio construction (add most divers

Simple example, entirely in sample, assumes means and variances are the same
Assumes equal weights

Required: pandas / numpy, matplotlib

USE AT YOUR OWN RISK! No warranty is provided or implied.

Handling of NAN's and Inf's isn't done here (except within pandas), 
And there is no error handling!

"""

#import datetime
#import pandas as pd
import numpy as np
from scipy.optimize import minimize
from common import pd_readcsv
from optimisation import correlation_matrix, neg_SR
from copy import copy

class instrument_index_list(object):
    """
    Manage the index of instruments we have to optimise over
    """
    
    def __init__(self, length_index):
        self.in_portfolio=[]
        self.not_in_portfolio=range(length_index)
        
    def add(self, index_to_pop):
        self.in_portfolio.append(index_to_pop)
        self.not_in_portfolio.remove(index_to_pop)
    
def return_sr(weights, data):
    """
    Telle me what the 
    
    In this simplified version we assume all assets have the same volatility and mean returns
    
    This means we only need to use the correlation matrix instead of the covariance
    And a dull vector of returns
    
    You can change this if required.... 
    """
    
    sigma=correlation_matrix(data)
    
    default_SR=1.0
    mus=np.array([(sigma.diagonal()[idx]**.5)*default_SR/16.0 for idx in range(len(data.columns))], ndmin=2).transpose()
    
    return -neg_SR(weights, sigma,mus)

def find_most_diversifying_asset(cmat):
    """
    Returns the most diversifying asset outright (lowest correlation) 
    """
    cavgs=avg_correlations(cmat)
    return np.argmin(cavgs)

def avg_correlations(cmat):
    """
    Return a list of the average correlations
    """
    new_cmat=copy(cmat)
    np.fill_diagonal(new_cmat, np.nan)
    avgs=np.nanmean(new_cmat, axis=0)
    
    return avgs

def set_weights(wts, indextoset):
    """
    Given a vector wts, set the values with index in indextoset to 1.0/(len(indextoset)), others to zero
    """
    
    weight=1.0/(len(indextoset))
    
    for idx in range(len(wts)):
        if idx in indextoset:
            wts[idx]=weight
        else:
            wts[idx]=0.0
            
    return wts
    
def find_best_addition(data, candidates, current_portfolio):
    """
    Iterate over candidates and return best
    """
    
    srlist=[]
    wts=[0.0]*(len(candidates)+len(current_portfolio))

    for candidx in candidates:
        cand_portfolio=current_portfolio+[candidx]
        wts=set_weights(wts, cand_portfolio)
        srlist.append(return_sr(wts, data))
        
    return candidates[np.argmax(srlist)]
    
    
if __name__=="__main__":
    
    ## Get the data
    
    """
    This is a trivial example, 3 possible assets, portfolio of 2
    
    """
    
    filename="assetprices.csv"
    data=pd_readcsv(filename)
    asset_names=data.columns
    
    assets_to_add=2
    
    cmat=correlation_matrix(data)
    
    instrument_index=instrument_index_list(data.shape[1])
    first_index=find_most_diversifying_asset(cmat)
    instrument_index.add(first_index)
    print "Started with %s" % asset_names[first_index] 
    
    while len(instrument_index.in_portfolio)<assets_to_add:
        current_portfolio=instrument_index.in_portfolio
        candidates=instrument_index.not_in_portfolio
        new_addition=find_best_addition(data, candidates, current_portfolio)
        print "Adding %s" % asset_names[new_addition]
        
        instrument_index.add(new_addition)
    
    print "Selected portfolio:"
    print [asset_names[xidx] for xidx in instrument_index.in_portfolio]