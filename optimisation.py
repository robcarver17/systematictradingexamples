"""
This file shows how to do a number of different optimisations - 'one shot' and bootstrapping ;
  also entirely in sample, expanding, and rolling windows 

As in chapters 3 and 4 of "Systematic Trading" by Robert Carver (www.systematictrading.org)

Required: pandas / numpy, matplotlib

USE AT YOUR OWN RISK! No warranty is provided or implied.

Handling of NAN's and Inf's isn't done here (except within pandas), 
And there is no error handling!

The bootstrapping method here is not 'block' bootstrapping, so any time series dependence of returns will be lost 

"""

import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from copy import copy
import random

HANDCRAFTED_WTS=pd.read_csv("handcraftweights.csv")

def pd_readcsv(filename):
    """
    Reads the pandas dataframe from a filename, given the index is correctly labelled
    """

    
    ans=pd.read_csv(filename)
    ans.index=pd.to_datetime(ans['DATETIME'])
    del ans['DATETIME']
    ans.index.name=None
    
    return ans

def correlation_matrix(returns):
    """
    Calcs a correlation matrix using weekly returns from a pandas time series

    We use weekly returns because otherwise end of day effects, especially over time zones, give
    unrealistically low correlations
    
    """
    asset_index=returns.cumsum().ffill()
    asset_index=asset_index.resample('1W') ## Only want index, fill method is irrelevant
    asset_index = asset_index - asset_index.shift(1)

    return asset_index.corr().values
    

def create_dull_pd_matrix(dullvalue=0.0, dullname="A", startdate=pd.datetime(1970,1,1).date(), enddate=datetime.datetime.now().date(), index=None):
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

def neg_SR_riskfree(weights, sigma, mus, riskfree=0.00):
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
    norm_returns=returns*facmat.values
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

def nearest_to_listvals(x, lvalues=[0.0, 0.25, 0.5, 0.75, 0.9]):
    ## return x rounded to nearest of lvalues
    
    if len(lvalues)==1:
        return lvalues[0]
    
    d1=abs(x - lvalues[0])
    d2=abs(x - lvalues[1])
    
    if d1<d2:
        return lvalues[0]
    
    newlvalues=lvalues[1:]
    
    return nearest_to_listvals(x, newlvalues)
    

def handcrafted(returns, equalisevols=True, default_vol=0.2):
    """
    Handcrafted optimiser
    """


    count_assets=len(returns.columns)
    
    try:
        
        assert equalisevols is True
        assert count_assets<=3
    except:
        raise Exception("Handcrafting only works with equalised vols and 3 or fewer assets")
    
    if count_assets<3:
        ## Equal weights
        return [1.0/count_assets]*count_assets

    est_corr=returns.corr().values
    c1=nearest_to_listvals(est_corr[0][1])
    c2=nearest_to_listvals(est_corr[0][2])
    c3=nearest_to_listvals(est_corr[1][2])
    
    wts_to_use=HANDCRAFTED_WTS[(HANDCRAFTED_WTS.c1==c1) & (HANDCRAFTED_WTS.c2==c2) & (HANDCRAFTED_WTS.c3==c3)].irow(0)

    return [wts_to_use.w1, wts_to_use.w2, wts_to_use.w3]

def opt_shrinkage(returns, shrinkage_factors, equalisevols=True, default_vol=0.2):
    """
    Returns the optimal portfolio for the dataframe returns using shrinkage

    shrinkage_factors is a tuple, shrinkage of mean and correlation
    
    If equalisevols=True then normalises returns to have same standard deviation; the weights returned
       will be 'risk weightings'
       
    
    """
    
    if equalisevols:
        use_returns=equalise_vols(returns, default_vol)
    else:
        use_returns=returns
    
    (shrinkage_mean, shrinkage_corr)=shrinkage_factors

    ## Sigma matrix 
    ## Use correlation and then convert back to variance
    est_corr=use_returns.corr().values
    avg_corr=get_avg_corr(est_corr)
    prior_corr=offdiag_matrix(avg_corr, est_corr.shape[0])

    sigma_corr=shrinkage_corr*prior_corr+(1-shrinkage_corr)*est_corr
    cov_vector=use_returns.std().values
    
    sigma=cov_vector*sigma_corr*cov_vector

    ## mus vector
    avg_return=np.mean(use_returns.mean())
    est_mus=np.array([use_returns[asset_name].mean() for asset_name in use_returns.columns], ndmin=2).transpose()
    prior_mus=np.array([avg_return for asset_name in use_returns.columns], ndmin=2).transpose()
    
    mus=shrinkage_mean*prior_mus+(1-shrinkage_mean)*est_mus
    
    ## Starting weights
    number_assets=use_returns.shape[1]
    start_weights=[1.0/number_assets]*number_assets
    
    ## Constraints - positive weights, adding to 1.0
    bounds=[(0.0,1.0)]*number_assets
    cdict=[{'type':'eq', 'fun':addem}]
    
    ans=minimize(neg_SR, start_weights, (sigma, mus), method='SLSQP', bounds=bounds, constraints=cdict, tol=0.00001)
        
    return ans['x']
    
    


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
    if equalisemeans:
        ## Don't use the data - Set to the average Sharpe Ratio
        avg_return=np.mean(use_returns.mean())
        mus=np.array([avg_return for asset_name in use_returns.columns], ndmin=2).transpose()

    else:
        mus=np.array([use_returns[asset_name].mean() for asset_name in use_returns.columns], ndmin=2).transpose()
    
    ## Starting weights
    number_assets=use_returns.shape[1]
    start_weights=[1.0/number_assets]*number_assets
    
    ## Constraints - positive weights, adding to 1.0
    bounds=[(0.0,1.0)]*number_assets
    cdict=[{'type':'eq', 'fun':addem}]
    
    ans=minimize(neg_SR, start_weights, (sigma, mus), method='SLSQP', bounds=bounds, constraints=cdict, tol=0.00001)
        
    return ans['x']




"""
Now we need to do this with expanding or rolling window
"""

"""
Generate the date tuples
"""

def generate_fitting_dates(data, date_method, rollyears=20):
    """
    generate a list 4 tuples, one element for each year in the data
    each tuple contains [fit_start, fit_end, period_start, period_end] datetime objects
    the last period will be a 'stub' if we haven't got an exact number of years
    
    date_method can be one of 'in_sample', 'expanding', 'rolling'
    
    if 'rolling' then use rollyears variable  
    """
    
    start_date=data.index[0]
    end_date=data.index[-1]
    
    ## generate list of dates, one year apart, including the final date
    yearstarts=list(pd.date_range(start_date, end_date, freq="12M"))+[end_date]
   
    ## loop through each period     
    periods=[]
    for tidx in range(len(yearstarts))[1:-1]:
        ## these are the dates we test in
        period_start=yearstarts[tidx]
        period_end=yearstarts[tidx+1]

        ## now generate the dates we use to fit
        if date_method=="in_sample":
            fit_start=start_date
        elif date_method=="expanding":
            fit_start=start_date
        elif date_method=="rolling":
            yearidx_to_use=max(0, tidx-rollyears)
            fit_start=yearstarts[yearidx_to_use]
        else:
            raise Exception("don't recognise date_method %s" % date_method)
            
        if date_method=="in_sample":
            fit_end=end_date
        elif date_method in ['rolling', 'expanding']:
            fit_end=period_start
        else:
            raise Exception("don't recognise date_method %s " % date_method)
        
        periods.append([fit_start, fit_end, period_start, period_end])

    ## give the user back the list of periods
    return periods

        
"""
Now do some fitting

Before we do that we need a bootstrap fitter
"""



def bootstrap_portfolio(returns_to_bs, monte_carlo=200, monte_length=250, equalisemeans=False, equalisevols=True, default_vol=0.2, default_SR=1.0):
    
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

def optimise_over_periods(data, date_method, fit_method, rollyears=20, equalisemeans=False, equalisevols=True, 
                          monte_carlo=200, monte_length=250, shrinkage_factors=(0.5, 0.5)):
    """
    Do an optimisation
    
    Returns data frame of weights
    
    Note if fitting in sample weights will be somewhat boring
    
    Doesn't deal with eg missing data in certain subperiods
    
    
    """
    
    ## Get the periods
    fit_periods=generate_fitting_dates(data, date_method, rollyears=rollyears)
    
    ## Do the fitting
    ## Build up a list of weights, which we'll concat
    weight_list=[]
    for fit_tuple in fit_periods:
        ## Fit on the slice defined by first two parts of the tuple
        period_subset_data=data[fit_tuple[0]:fit_tuple[1]]
        
        ## Can be slow, if bootstrapping, so indicate where we are
        
        print "Fitting data for %s to %s" % (str(fit_tuple[2]), str(fit_tuple[3]))
        
        if fit_method=="one_period":
            weights=markosolver(period_subset_data, equalisemeans=equalisemeans, equalisevols=equalisevols)
        elif fit_method=="bootstrap":
            weights=bootstrap_portfolio(period_subset_data, equalisemeans=equalisemeans, 
                                        equalisevols=equalisevols, monte_carlo=monte_carlo, 
                                        monte_length=monte_length)
        elif fit_method=="shrinkage":
            weights=opt_shrinkage(period_subset_data, shrinkage_factors=shrinkage_factors, equalisevols=equalisevols)
            
        else:
            raise Exception("Fitting method %s unknown" % fit_method)
        
        ## We adjust dates slightly to ensure no overlaps
        dindex=[fit_tuple[2]+datetime.timedelta(seconds=1), fit_tuple[3]-datetime.timedelta(seconds=1)]
        
        ## create a double row to delineate start and end of test period
        weight_row=pd.DataFrame([weights]*2, index=dindex, columns=data.columns)
        
        weight_list.append(weight_row)
        
    weight_df=pd.concat(weight_list, axis=0)
    
    return weight_df


def opt_and_plot(*args, **kwargs):
    """
    Function to do optimisation and plotting in one go

    """

    mat1=optimise_over_periods(*args, **kwargs)
    mat1.plot()
    plt.show()


## Get the data

filename="assetprices.csv"
data=pd_readcsv(filename)

## Let's do some optimisation
## Feel free to play with these

if __name__=="__main__":
    
    """
    Remember the arguments are:
    data, date_method, fit_method, rollyears=20, equalisemeans=False, equalisevols=True, 
                              monte_carlo=200, monte_length=250
    
    
    opt_and_plot(data, "in_sample", "one_period", equalisemeans=False, equalisevols=False)
    
    opt_and_plot(data, "in_sample", "one_period", equalisemeans=False, equalisevols=True)
    
    opt_and_plot(data, "in_sample", "one_period", equalisemeans=True, equalisevols=True)
    
    opt_and_plot(data, "in_sample", "bootstrap", equalisemeans=False, equalisevols=True, monte_carlo=500)
    
    opt_and_plot(data, "rolling", "one_period", rollyears=1, equalisemeans=False, equalisevols=True)
    
    opt_and_plot(data, "rolling", "one_period", rollyears=5, equalisemeans=False, equalisevols=True)
    
    opt_and_plot(data, "expanding", "one_period", equalisemeans=False, equalisevols=True)
    
    opt_and_plot(data, "expanding", "bootstrap", equalisemeans=False, equalisevols=True)
    
    """