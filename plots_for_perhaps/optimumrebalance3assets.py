from optimisation import *
from matplotlib.pyplot import plot, show, xticks, xlabel, ylabel, legend, yscale, title, savefig, rcParams, figure, hist, text, bar, subplots
import matplotlib.pyplot as plt
import Image

from itertools import cycle

from datetime import datetime as dt
from twisted.test.test_amp import THING_I_DONT_UNDERSTAND


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, show, xticks, xlabel, ylabel, legend, yscale, title, savefig, rcParams, figure, hist, text, bar, subplots
import Image

def file_process(filename):
    fig = plt.gcf()
    fig.set_size_inches(18.5,10.5)
    fig.savefig("/home/rob/%s.png" % filename,dpi=300)
    fig.savefig("/home/rob/%sLOWRES.png" % filename,dpi=50)
    
    Image.open("/home/rob/%s.png" % filename).convert('L').save("/home/rob/%s.jpg" % filename)
    Image.open("/home/rob/%sLOWRES.png" % filename).convert('L').save("/home/rob/%sLOWRES.jpg" % filename)


lines = ["-","--","-."]
linecycler = cycle(lines)    
colorcycler=cycle(["red", "blue", "green"])

def uniquets(df3):
    """
    Makes x unique
    """
    df3=df3.groupby(level=0).first()
    return df3


def read_ts_csv(fname, dindex="Date"):
    data=pd.read_csv(fname)
    dateindex=[dt.strptime(dx, "%d/%m/%y") for dx in list(data[dindex])]
    data.index=dateindex
    del(data[dindex])
    
    return data

def calc_asset_returns(rawdata, tickers):
    asset_returns=pd.concat([get_monthly_tr(tickname, rawdata) for tickname in tickers], axis=1)
    asset_returns.columns=tickers

    return asset_returns

def get_monthly_tr(tickname, rawdata):
    total_returns=rawdata[tickname+"_TR"]
    return (total_returns / total_returns.shift(1)) - 1.0

def get_div_yield(tickname, rawdata):
    price_data=rawdata[tickname+"_PRICE"]
    total_returns=rawdata[tickname+"_TR"]

    tr_return = get_monthly_tr(tickname, rawdata)
    price_return = (price_data / price_data.shift(1)) - 1.0
    div = tr_return - price_return
    div_tr = div * total_returns
    last_year_divs = pd.rolling_mean(div_tr, 12, min_periods=1)*12.0
    div_yield = last_year_divs / total_returns
    
    return div_yield


SRdifftable=[     -0.25, -0.2, -0.15,  -0.1, -0.05, 0.0, 0.05, 0.1,  0.15, 0.2, 0.25]
multipliers=[.6,    .66, .77,     .85,  .94,   1.0, 1.11, 1.19, 1.3, 1.37, 1.48]

def risk_weights_with_model(rowidx, forecast, match_rawweights, SRdifftable, multipliers):
    
    start_weights=list(match_rawweights.irow(rowidx).values)
    SRdifflist = list(forecast.irow(rowidx).values)
    
    def process(SRdiff):
        
        if np.isnan(SRdiff):
            return 1.0
        
        if SRdiff>0.25:
            SRdiff=0.25
        if SRdiff<-0.25:
            SRdiff=-0.25
        
        return np.interp(SRdiff, SRdifftable, multipliers)
    
    new_risk_weights=[process(SRdiff)*start_wt for SRdiff, start_wt in zip(SRdifflist, start_weights)]
    new_risk_weights=[wt/sum(new_risk_weights) for wt in new_risk_weights]
    
    return new_risk_weights

def yield_forecast(data, vols):
    
    normyields = yields/vols

    avgyield= normyields.mean(axis=1)
    avgyield = pd.DataFrame(np.array([avgyield]*len(data.columns), ndmin=2).transpose(), data.index)
    avgyield.columns=normyields.columns   
    
    diff = normyields - avgyield
    
    return diff*3.0


def trailing_forecast( data, vols):
    
    annindex=data.cumsum()
    annreturns=annindex - annindex.shift(12)
    normreturns=annreturns/vols    
    
    avgreturn = normreturns.mean(axis=1)
    avgreturn = pd.DataFrame(np.array([avgreturn]*len(data.columns), ndmin=2).transpose(), data.index)
    avgreturn.columns=normreturns.columns
    
    diff = normreturns - avgreturn
    
    return diff*.5


def calc_ts_new_risk_weights(data, vols, rawweights, SRdifftable, multipliers, forecast_func):
    
    forecast=forecast_func( data, vols)
    match_rawweights=rawweights.reindex(forecast.index, method="ffill")
    
    new_weights=[risk_weights_with_model(rowidx, forecast, match_rawweights, SRdifftable, multipliers) for 
                 rowidx in range(len(forecast))]
    
    return pd.DataFrame(new_weights, match_rawweights.index, columns=match_rawweights.columns)

def get_equities_data(case="devall"):
    rawdata=read_ts_csv("/home/rob/workspace/systematictradingexamples/plots_for_perhaps/MSCI_data.csv")
    refdata=pd.read_csv("/home/rob/workspace/systematictradingexamples/plots_for_perhaps/MSCI_ref.csv")
    if case=="emdev":
        tickers=["EM", "DEV"] #mom 9bp
    elif case=="devregions":
        tickers=list(refdata[(refdata.EmorDEV=="DEV") & (refdata.Type=="Region")].Country.values) #mom 2sr
    elif case=="emregions":
        tickers=list(refdata[(refdata.EmorDEV=="EM") & (refdata.Type=="Region")].Country.values) #divs 3sr
    elif case=="devemea":
        tickers=list(refdata[(refdata.EmorDEV=="DEV") & (refdata.Region=="EMEA")].Country.values) #mom
    elif case=="devams":
        tickers=list(refdata[(refdata.EmorDEV=="DEV") & (refdata.Region=="America")].Country.values) #mom

    elif case=="devasia":
        tickers=list(refdata[(refdata.EmorDEV=="DEV") & (refdata.Region=="Asia")].Country.values) #divs
    elif case=="emasia":
        tickers=list(refdata[(refdata.EmorDEV=="EM") & (refdata.Region=="Asia")].Country.values) #flat
    elif case=="emams":
        tickers=list(refdata[(refdata.EmorDEV=="EM") & (refdata.Region=="America")].Country.values) #divs
    elif case=="ememea":
        tickers=list(refdata[(refdata.EmorDEV=="EM") & (refdata.Region=="EMEA")].Country.values) #divs
    elif case=="devall":
        tickers=list(refdata[(refdata.EmorDEV=="DEV") & (refdata.Type=="Country")].Country.values) #mom 12bp
        hc_risk_weights=pd.read_csv("/home/rob/workspace/systematictradingexamples/plots_for_perhaps/devhcweights.csv")
        initial_risk_weights=[hc_risk_weights[hc_risk_weights.Country==code].Weight.values[0] for code in tickers]
        
    elif case=="emall":
        tickers=list(refdata[(refdata.EmorDEV=="EM") & (refdata.Type=="Country")].Country.values) #mom 7bp
    elif case=="all":
        tickers=list(refdata[refdata.Type=="Country"].Country.values) #mom 17bp
    elif case=="emea":
        tickers=list(refdata[refdata.Region=="EMEA"].Country.values) #mom
    elif case=="america":
        tickers=list(refdata[refdata.Region=="America"].Country.values) #mom
    elif case=="asia":
        tickers=list(refdata[refdata.Region=="Asia"].Country.values) #flat
    else:
        raise Exception()

    def vl(emordev):
        if emordev=="EM":
            return .23
        else:
            return .15

    data=calc_asset_returns(rawdata, tickers)

    yields=pd.concat([get_div_yield(tickname, rawdata) for tickname in tickers], axis=1, names=tickers)
    yields.columns=tickers
        
    initial_stdev=[vl(refdata[refdata.Country==ticker].EmorDEV.values[0]) for ticker in tickers]

    

    return data, initial_stdev, initial_risk_weights, yields


def get_some_crossasset_data():

    rawdata=read_ts_csv("/home/rob/workspace/systematictradingexamples/plots_for_perhaps/MSCI_data.csv")
    
    data=pd.read_csv("/home/rob/workspace/systematictradingexamples/plots_for_perhaps/USmonthlyreturns.csv")
    
    data = data[1190:]
    
    
    
    dindex="Date"
    dateindex=[dt.strptime(dx, "%d/%m/%Y") for dx in list(data[dindex])]
    data.index=dateindex
    del(data[dindex])
    
    USeqyield = data.SP_Yield
    USbondyield = data.Bond_Yield


    asset_returns2=calc_asset_returns(rawdata, ["UK"])
    asset_returns2.columns=["UK_Equity"]
    
    UKequityyield=get_div_yield("UK", rawdata)*100.0
    
    asset_returns=data[["SP_TR", "Bond_TR"]]
    asset_returns.columns=["US_Equity", "US_Bond"]
    
    asset_returns.index = [x + pd.DateOffset(months=1) for x in asset_returns.index]
    asset_returns.index = [x - pd.DateOffset(days=1) for x in asset_returns.index]
    
    
    data=pd.concat([asset_returns, asset_returns2], axis=1)
    data = data.cumsum().ffill().reindex(asset_returns.index).diff()
    
    data.columns=["US Equity" ,"US Bond", "UK Equity"]
    
    yields=pd.concat([USeqyield, USbondyield, UKequityyield], axis=1)
    yields=yields.ffill().reindex(data.index).ffill()
    yields.columns=["US Equity" ,"US Bond", "UK Equity"]

    yields = yields/100.0

    initial_stdev=[0.222,.08,.222]
    initial_risk_weights=[.25,.5,.25]



    return data, initial_stdev, initial_risk_weights, yields




## we ignore fx rates
## prices will have more impact anyway

def calc_cash_weights_from_risk_weight(risk_weights, stdevlist):
    cash_weights=[x/y for (x,y) in zip(risk_weights,stdevlist)]

    total_cash=np.sum(cash_weights)
    cash_weights=[x/total_cash for x in cash_weights]

    return cash_weights

def calc_shares_from_cash_weight(cash_weights, account_value, prices):
    cash_value=[account_value*y for y in cash_weights]
    shares=[round(x/y) for (x,y) in zip(cash_value, prices)]
    return shares

def calc_shares_from_risk_weight(risk_weights, stdevlist, account_value, prices):
    cash_weights=calc_cash_weights_from_risk_weight(risk_weights, stdevlist)
    shares=calc_shares_from_cash_weight(cash_weights, account_value, prices)
    return shares
    
calc_shares_from_risk_weight([.25,.25,.5], [.2,.15,.22],10000, [2,5, (30/1.5)])

def implied_cash_weights(numb_shares, prices, account_value):
    return [float(y)*float(z)/account_value for (y,z) in zip(numb_shares, prices)]

def risk_weights_implied_by(numb_shares, prices, account_value, stdevlist):
    cash_weights=implied_cash_weights(numb_shares, prices, account_value)
    risk_weights=[x*y for (x,y) in zip(cash_weights,stdevlist)]
    
    total_cash=np.sum(cash_weights)
    total_risk=np.sum(risk_weights)

    return [total_cash * x / total_risk for x in risk_weights]


risk_weights_implied_by([1204,642,218], [2.0,5.0, (30/1.5)], 10000, [.2,.15,.22])

### Start with simple, constant risk weights
### process: each period recalculate portfolio
### check to see if rebalance needed

START_PRICE=10.0

def calc_initial_share_count( start_portfolio_value, risk_weights=[], stdevlist=[] 
                             ):
    start_prices=[START_PRICE]*len(risk_weights) ## by construction
    return calc_shares_from_risk_weight(risk_weights, stdevlist, start_portfolio_value, start_prices)
    
    

tradecosts_US=[[100,                       1000,  5000,      10000,          1000000],
               [1.1,                      1.5,    3.5,       6.,           2500.0]]
tradecosts_UK=[[  100, 1000,5000,        10000,          1000000],
               [ 6.3,  9,  21.0,         36.,            5100.0]]

## Now ...
def cost_trade(value, country="US"):
    if value==0:
        return 0.0
    if country=="US":
        tradecosts=tradecosts_US
    elif country=="UK":
        tradecosts=tradecosts_UK
    cost=np.interp(value, tradecosts[0], tradecosts[1], right=-1)
    if cost==-1:
        ## use same %
        cost=value*(tradecosts[1][-1]/tradecosts[0][-1])
    return cost


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
    yearstarts=list(pd.date_range(start_date, end_date, freq="12M"))
    if yearstarts[-1]<end_date:
        yearstarts=yearstarts+[end_date]
   
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

import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from copy import copy
import random

FLAG_BAD_RETURN=0.0000001

def fix_mus(mean_list):
    """
    Replace nans with unfeasibly large negatives
    
    result will be zero weights for these assets
    """
    
    def _fixit(x):
        if np.isnan(x):
            return FLAG_BAD_RETURN
        else:
            return x
    
    mean_list=[_fixit(x) for x in mean_list]
    
    return mean_list

def un_fix_weights(mean_list, weights):
    """
    When mean has been replaced, use nan weight
    """
    
    def _unfixit(xmean, xweight):
        if xmean==FLAG_BAD_RETURN:
            return 0.0
        elif xmean<0.0:
            return 0.0
            
        return xweight
    
    fixed_weights=[_unfixit(xmean, xweight) for (xmean, xweight) in zip(mean_list, weights)]
    
    return fixed_weights


def fix_sigma(sigma):
    """
    Replace nans with zeros
    
    """
    
    def _fixit(x):
        if np.isnan(x):
            return 0.0
        else:
            return x
    
    sigma=[[_fixit(x) for x in sigma_row] for sigma_row in sigma]
    
    sigma=np.array(sigma)
    
    return sigma



def must_have_item(slice_data):
    """
    Returns the columns of slice_data for which we have at least one non nan value  
    
    :param slice_data: Data to get correlations from
    :type slice_data: pd.DataFrame

    :returns: list of bool

    >>>
    """
        
    def _any_data(xseries):
        data_present=[not np.isnan(x) for x in xseries]
        
        return any(data_present)
    
    some_data=slice_data.apply(_any_data, axis=0)
    some_data_flags=list(some_data.values)
    
    return some_data_flags


def optimise( sigma, mean_list):
    
    ## will replace nans with big negatives
    mean_list=fix_mus(mean_list)
    
    ## replaces nans with zeros
    sigma=fix_sigma(sigma)
    
    mus=np.array(mean_list, ndmin=2).transpose()
    number_assets=sigma.shape[1]
    start_weights=[1.0/number_assets]*number_assets
    
    ## Constraints - positive weights, adding to 1.0
    bounds=[(0.0,1.0)]*number_assets
    cdict=[{'type':'eq', 'fun':addem}]
    ans=minimize(neg_SR, start_weights, (sigma, mus), method='SLSQP', bounds=bounds, constraints=cdict, tol=0.00001)

    ## anything that had a nan will now have a zero weight
    weights=ans['x']
    
    ## put back the nans
    weights=un_fix_weights(mean_list, weights)
    
    return weights

def clean_weights(weights,  must_haves=None, fraction=0.5):
    """
    Make's sure we *always* have some weights where they are needed, by replacing nans
    Allocates fraction of pro-rata weight equally
    
    :param weights: The weights to clean
    :type weights: list of float

    :param must_haves: The indices of things we must have weights for
    :type must_haves: list of bool

    :param fraction: The amount to reduce missing instrument weights by
    :type fraction: float

    :returns: list of float

    >>> clean_weights([1.0, np.nan, np.nan],   fraction=1.0)
    [0.33333333333333337, 0.33333333333333331, 0.33333333333333331]
    >>> clean_weights([0.4, 0.6, np.nan],  fraction=1.0)
    [0.26666666666666672, 0.40000000000000002, 0.33333333333333331]
    >>> clean_weights([0.4, 0.6, np.nan],  fraction=0.5)
    [0.33333333333333337, 0.5, 0.16666666666666666]
    >>> clean_weights([np.nan, np.nan, 1.0],  must_haves=[False,True,True], fraction=1.0)
    [0.0, 0.5, 0.5]
    >>> clean_weights([np.nan, np.nan, np.nan],  must_haves=[False,False,True], fraction=1.0)
    [0.0, 0.0, 1.0]
    >>> clean_weights([np.nan, np.nan, np.nan],  must_haves=[False,False,False], fraction=1.0)
    [0.0, 0.0, 0.0]
    """
    ### 

    if must_haves is None:
        must_haves=[True]*len(weights)
    
    if not any(must_haves):
        return [0.0]*len(weights)
    
    needs_replacing=[(np.isnan(x) or x==0.0) and must_haves[i] for (i,x) in enumerate(weights)]
    keep_empty=[(np.isnan(x) or x==0.0) and not must_haves[i] for (i,x) in enumerate(weights)]
    no_replacement_needed=[(not keep_empty[i]) and (not needs_replacing[i]) for (i,x) in enumerate(weights)]

    if not any(needs_replacing):
        return weights
    
    missing_weights=sum(needs_replacing)

    total_for_missing_weights=fraction*missing_weights/(
        float(np.nansum(no_replacement_needed)+np.nansum(missing_weights)))
    
    adjustment_on_rest=(1.0-total_for_missing_weights)
    
    each_missing_weight=total_for_missing_weights/missing_weights
    
    def _good_weight(value, idx, needs_replacing, keep_empty, 
                     each_missing_weight, adjustment_on_rest):
            
        if needs_replacing[idx]:
            return each_missing_weight
        if keep_empty[idx]:
            return 0.0
        if value<0.0:
            return 0.0
        return value*adjustment_on_rest

    weights=[_good_weight(value, idx, needs_replacing, keep_empty, 
                          each_missing_weight, adjustment_on_rest) 
             for (idx, value) in enumerate(weights)]
    
    ## This process will screw up weights - let's fix them
    xsum=sum(weights)
    weights=[x/xsum for x in weights]
    
    return weights


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
    sigma=sigma_from_corr(std, corr)
    
    ans=optimise(sigma, list(mus))
    
    return ans

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


def opt_shrinkage(returns, shrinkage_factors, equalisevols=True, default_vol=0.2, must_haves=[]):
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
    sigma=fix_sigma(sigma)


    ## mus vector
    avg_return=np.mean(use_returns.mean())
    est_mus=np.array([use_returns[asset_name].mean() for asset_name in use_returns.columns], ndmin=2).transpose()
    prior_mus=np.array([avg_return for asset_name in use_returns.columns], ndmin=2).transpose()
    
    mus=shrinkage_mean*prior_mus+(1-shrinkage_mean)*est_mus

    mean_list=list(mus.transpose()[0])
    mean_list=fix_mus(mean_list)
    mus=np.array(mean_list, ndmin=2).transpose()

    
    ## Starting weights
    number_assets=use_returns.shape[1]
    start_weights=[1.0/number_assets]*number_assets
    
    ## Constraints - positive weights, adding to 1.0
    bounds=[(0.0,1.0)]*number_assets
    cdict=[{'type':'eq', 'fun':addem}]
    
    ans=minimize(neg_SR, start_weights, (sigma, mus), method='SLSQP', bounds=bounds, constraints=cdict, tol=0.00001)
    weights=ans['x']
    weights=un_fix_weights(mean_list, weights)
    
    weights=clean_weights(weights, must_haves)
        
    return weights

def markosolver(returns, equalisemeans=False, equalisevols=True, default_vol=0.2, default_SR=1.0,
                must_haves=[]):
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
    sigma=fix_sigma(sigma)

    ## Expected mean returns    
    if equalisemeans:
        ## Don't use the data - Set to the average Sharpe Ratio
        avg_return=np.mean(use_returns.mean())
        mus=[avg_return for asset_name in use_returns.columns]
        
        mus=np.array(mus, ndmin=2).transpose()
        mus[np.isnan(use_returns.mean().values)]=np.nan

    else:
        mus=np.array([use_returns[asset_name].mean() for asset_name in use_returns.columns], ndmin=2).transpose()

    mean_list=list(mus.transpose()[0])
    mean_list=fix_mus(mean_list)
    mus=np.array(mean_list, ndmin=2).transpose()

    
    ## Starting weights
    number_assets=use_returns.shape[1]
    start_weights=[1.0/number_assets]*number_assets
    
    ## Constraints - positive weights, adding to 1.0
    bounds=[(0.0,1.0)]*number_assets
    cdict=[{'type':'eq', 'fun':addem}]
    
    ans=minimize(neg_SR, start_weights, (sigma, mus), method='SLSQP', bounds=bounds, constraints=cdict, tol=0.00001)
    weights=ans['x']    

    weights=un_fix_weights(mean_list, weights)
    
    weights=clean_weights(weights, must_haves)

    return weights


def bootstrap_portfolio(returns_to_bs, monte_carlo=200, monte_length=250, equalisemeans=False, equalisevols=True, default_vol=0.2, default_SR=1.0,
                        must_haves=[]):
    
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
        weight=markosolver(returns, equalisemeans=equalisemeans, equalisevols=equalisevols, default_vol=default_vol, default_SR=default_SR,
                           must_haves=must_haves)
        weightlist.append(weight)
        
    ### We can take an average here; only because our weights always add up to 1. If that isn't true
    ###    then you will need to some kind of renormalisation
     
    theweights_mean=list(np.mean(weightlist, axis=0))
    
    theweights_mean=[x/sum(theweights_mean) for x in theweights_mean]
    
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

        current_period_data=data[fit_tuple[2]:fit_tuple[3]]
        must_haves=must_have_item(current_period_data)

        
        if fit_method=="one_period":
            weights=markosolver(period_subset_data, equalisemeans=equalisemeans, equalisevols=equalisevols, must_haves=must_haves)
        elif fit_method=="bootstrap":
            weights=bootstrap_portfolio(period_subset_data, equalisemeans=equalisemeans, 
                                        equalisevols=equalisevols, monte_carlo=monte_carlo, 
                                        monte_length=monte_length, must_haves=must_haves)
        elif fit_method=="shrinkage":
            weights=opt_shrinkage(period_subset_data, shrinkage_factors=shrinkage_factors, equalisevols=equalisevols, 
                                  must_haves=must_haves)
            
        else:
            raise Exception("Fitting method %s unknown" % fit_method)
        
        ## We adjust dates slightly to ensure no overlaps
        dindex=[fit_tuple[2]]
        
        ## create a double row to delineate start and end of test period
        weight_row=pd.DataFrame([weights], index=dindex, columns=data.columns)
        
        weight_list.append(weight_row)
        
    weight_df=pd.concat(weight_list, axis=0)
    
    return weight_df



def get_current_prices(idx, priceindex):
    return list(priceindex.irow(idx).values)

def get_monthly_divis(idx, numb_shares, yields, current_prices):
    current_yield=list(yields.irow(idx).values)
    return np.sum([(x*y*z)/1200.0 for x,y,z in zip(current_yield, numb_shares, current_prices)])

def get_account_value(current_prices, cash, numb_shares):
    return np.sum([x*y for (x,y) in zip(current_prices, numb_shares)])+cash

def calc_trade_size(numb_shares, current_optimal_risk_weights, stdevlist, current_account_value, current_prices, windowsize1, windowsize2):
    windowsize1a=windowsize1/2.0
    windowsize2a=windowsize2/2.0
    current_cash_weights=implied_cash_weights(numb_shares, current_prices, current_account_value)
    optimal_cash_weights=calc_cash_weights_from_risk_weight(current_optimal_risk_weights, stdevlist)
    differences= [x-y for x,y in zip(optimal_cash_weights,current_cash_weights)]
    #prop_differences= [(x/y)-1 for x,y in zip(optimal_cash_weights,current_cash_weights)]

    ## positive means buy

    sparecash=sum(differences) ## positive means spare cash
    
    #bigenoughdifferences=[abs(x)>windowsize for x in prop_differences]
    bigenoughdifferences=[abs(x)>(windowsize1a+windowsize2a) for x in differences]
    
    if not any(bigenoughdifferences):
        return [0.0]*len(numb_shares)

    def sizeit(x, windowsize1a):
        if x>0:
            return x - windowsize1a
        else:
            return windowsize1a + x
    
    difflist=[sizeit(x, windowsize1a) for idx, x in enumerate(differences) if bigenoughdifferences[idx]]
    
    nettrade=sum(difflist)
    
    ## if nettrade>0 want to buy, if sparecash<0 have to sell
    ## if nettrade>0 want to buy, if sparecash<nettrade have to buy less
    ## if nettrade<0 want to sell, if sparecash>0 can do so
    ## if nettrade<0 want to sell, if sparecash<nettrade have to sell more
    
    desirednet= min(sparecash,nettrade) ## spare cash means can buy more
    avgnet=desirednet/len(difflist)
    
    def _did(bigenough, diffvalue, avgnet):
        if bigenough:
            return diffvalue+avgnet
        return 0.0
    
    diffstoapply=[_did(bigenough, diffvalue, avgnet) for bigenough, diffvalue in zip(bigenoughdifferences, differences)]
    
    cash_weights_to_trade_to=[x+y for x,y in zip(current_cash_weights, diffstoapply)]
    
    correct_share_count=calc_shares_from_cash_weight(cash_weights_to_trade_to, current_account_value, current_prices)
    
    def tradeit(bigenough, correctshares, currentshares):
        if bigenough:
            return correctshares - currentshares
        return 0.0
    
    tradelist=[x-y for (x,y) in zip(correct_share_count, numb_shares)]

    
    return tradelist

## graphs showing

## first graph:
##  pick on as example. For a given country and setup and account size (usevaryingvol=False, fixedriskweights)

## for different setup [country, portfolio value, 

    
start_portfolio_value=100000
country="US"
yearband=5 ## blocks to divide data into
    


#datatype="Three_assets"
datatype="Developed_equities"

if datatype=="Three_assets":
    data, initial_stdev, initial_risk_weights, yields = get_some_crossasset_data()
elif datatype=="Developed_equities":
    data, initial_stdev, initial_risk_weights, yields = get_equities_data(case="devall")
else:
    raise Exception()
## rolling stdev for estimates if used
stdevest=pd.ewmstd(data, halflife=12)*(12**.5)
for idx in range(12):
    stdevest.iloc[idx,:]=initial_stdev

stdevest[stdevest==0]=np.nan

#bootstrapped_weights=optimise_over_periods(data, "rolling", "bootstrap", rollyears=5, equalisemeans=True, equalisevols=True, 
#                                  monte_carlo=50, monte_length=12*5)

bootstrapped_weights=optimise_over_periods(data, "rolling", "shrinkage", rollyears=5,  equalisevols=True, 
                                  shrinkage_factors=(1.0, 0.8))


avg_size=1.0/len(data.columns)

#windowsizes=[0.5]
windowsizes=[0.0, .05,         .1, .25  ,.5, .75, 1.0, 2.0]
windowsizelist=list(avg_size*np.array(windowsizes))


#min_sizes=[0.2500]
#min_sizes=[0.0,  250.0, 500.0]
min_sizes=[0.0, 100.0, 250.0, 500.0, 750.0, 1000.0]



    
start_date=data.index[0]
end_date=data.index[-1]

## generate list of dates, one year apart, including the final date
yearstarts=pd.date_range(data.index[0], data.index[-12*yearband], freq="12M")
yearends=yearstarts+pd.DateOffset(years=yearband)

yearstarts=list(yearstarts)
yearends=list(yearends)

fixed_weights=pd.DataFrame([initial_risk_weights]*len(data), data.index)
fixed_weights.columns=data.columns
fixed_weights[np.isnan(data)]=0.0
sumweights=list(fixed_weights.sum(axis=1).values)
sumweights=pd.DataFrame(np.array([sumweights]*len(data.columns), ndmin=2).transpose(), fixed_weights.index, data.columns)
sumweights.columns=fixed_weights.columns
fixed_weights=fixed_weights/sumweights


## parameters: usevaryingvol, usebootstrapweights, usemodel
parameter_set_names=["Baseline", "Varying vol", "Use correlations", "Yield model", "Momentum model",
                     "Vol + correlation",
                     "Vol + correlation + momentum", "Vol + correlation + yield"]

parameter_set=[(False, False, "None"), (True, False, "None"), 
               (False, True, "None"), (False, False, "yields"), (False, False, "momentum"), 
               (True, True, "None"),
               (True, True, "momentum"), (True, True, "yields")]


really_big_fat=[]

for min_trade_size in min_sizes:
    windowsize2=min_trade_size / start_portfolio_value    

    big_fat_results_dict=dict(turnover=dict(), netreturn=dict(), costs=dict(), percentile_gross=dict(),
                              percentile_net=dict(), grossreturn=dict())

    #for par_name, unpicked_parameters in zip(parameter_set_names, parameter_set):
    for par_name, unpicked_parameters in zip(['Baseline'], [(True, True, "momentum")]):
    #for par_name, unpicked_parameters in zip(['Baseline'], [(False, False, "None")]):
    
        usevaryingvol, usebootstrapsweights, usemodel = unpicked_parameters 
    
        if usevaryingvol:
            vols=stdevest
        else:
            vols=pd.DataFrame([initial_stdev]*len(data), data.index)
            vols.columns=data.columns
            
        if usebootstrapsweights:
            rawweights=copy(bootstrapped_weights)
        else:
            rawweights=copy(fixed_weights)
        
        if usemodel is "yields":
            weights=calc_ts_new_risk_weights(data, vols, rawweights, SRdifftable, multipliers, yield_forecast) 
        
        elif usemodel is "momentum":
            weights=calc_ts_new_risk_weights(data, vols, rawweights, SRdifftable, multipliers, trailing_forecast) 
            
        elif usemodel is "None":
            weights=copy(rawweights)
        
        ## assume monthly dividends
        ## it's unlikely to matter in practice
        
        ## any subsetting should be done here
        big_fat_results_dict['turnover'][par_name]=[]
        big_fat_results_dict['grossreturn'][par_name]=[]
        big_fat_results_dict['netreturn'][par_name]=[]
        big_fat_results_dict['costs'][par_name]=[]
        big_fat_results_dict['percentile_gross'][par_name]=[]
        big_fat_results_dict['percentile_net'][par_name]=[]
        
        for windowsize1 in windowsizelist:
            results=dict(turnover=[], grossreturn=[], netreturn=[], costs=[])
                
            for (start_date, end_date) in zip(yearstarts, yearends):
            
                print ("%s %.2f %.2f %s %s " % (par_name, windowsize1, windowsize2, str(start_date), str(end_date)))
        
                use_data=data[start_date:end_date]
                
                useyields=yields.reindex(use_data.index).ffill()
                usestdev=vols.reindex(use_data.index).ffill()
                useweights=uniquets(weights).reindex(use_data.index, method="ffill")
                
                pricereturns= use_data - useyields/1200.0
                priceindex=(pricereturns+1.0)
                priceindex=priceindex.cumprod()*START_PRICE
        
                ann_time_factor=12.0/len(priceindex.index)
                
                numb_shares=calc_initial_share_count(start_portfolio_value, risk_weights=list(useweights.irow(0).values), stdevlist=list(vols.irow(0).values), 
                                             )
                cash=start_portfolio_value - get_account_value([START_PRICE]*len(priceindex.columns), 0.0, numb_shares)
                
                ## loop
                
                idx=1
                
                accvalue=[start_portfolio_value]
                costs=[0.0]
                turnover=[0.0]
                
                while idx<(len(priceindex.index)):
                    stdevlist=list(usestdev.irow(idx).values)
                        
                    current_optimal_risk_weights = list(useweights.irow(idx).values)
                        
                    
                    current_prices=get_current_prices(idx, priceindex)
                    
                    this_months_divis=get_monthly_divis(idx, numb_shares, useyields, current_prices)
                    
                    
                    current_account_value=get_account_value(current_prices, cash, numb_shares)
                    
                    shares_traded = calc_trade_size(numb_shares, current_optimal_risk_weights, 
                                                    stdevlist, current_account_value, current_prices, 
                                                    windowsize1, windowsize2) 
                    
                    
                    numb_shares = [x+y for x,y in zip(shares_traded, numb_shares)]
                    
                    ## Note - this thing will compound like crazy so need to take 5 year snapshots
                    ## This is also a good thing with respect to averaging
                    
                    value_traded = [(x*y) for x,y in zip(shares_traded, current_prices)]
                    
                    commissions=np.sum([cost_trade(abs(x*y), country) for x,y in zip(shares_traded, current_prices)])
                    
                    cash = cash + this_months_divis + (-np.sum(value_traded))  - commissions
                    
                    accvalue.append(current_account_value)
                    costs.append(commissions)
                    turnover.append(np.sum([abs(x) for x in value_traded]))
                    
                    #print ("Account value %.1f Cash %.1f" % (current_account_value, cash))
                        
                    idx=idx+1
                
                ## all of these need annualising
                
                finalvalue=(accvalue[-1]/start_portfolio_value)-1.0
                totalcosts=np.sum(costs)/start_portfolio_value
                totalturnover=np.sum(turnover)/(start_portfolio_value)
                
                
                annual_turnover = totalturnover*ann_time_factor
                annual_return=((finalvalue+1.0)**ann_time_factor)-1.0
                annual_costs = totalcosts *ann_time_factor
                
                results['turnover'].append(annual_turnover) 
                results['grossreturn'].append(annual_return+annual_costs)
                results['netreturn'].append(annual_return)
                results['costs'].append(annual_costs)
                                            
            ## 75% Rule
            ## 75% of gross return; less costs
            
            avg_turnover=np.nanmean(results['turnover'])
            avg_gross=np.nanmean(results['grossreturn'])
            avg_net=np.nanmean(results['netreturn'])
            avg_costs=np.nanmean(results['costs'])
            percentile_gross=np.percentile(results['grossreturn'],25.0)
            percentile_net=percentile_gross - avg_costs
            
            big_fat_results_dict['turnover'][par_name].append(avg_turnover)    
            big_fat_results_dict['grossreturn'][par_name].append(avg_gross)    
            big_fat_results_dict['netreturn'][par_name].append(avg_net)    
            big_fat_results_dict['costs'][par_name].append(avg_costs)    
            big_fat_results_dict['percentile_gross'][par_name].append(percentile_gross)
            big_fat_results_dict['percentile_net'][par_name].append(percentile_net)

    really_big_fat.append(big_fat_results_dict)
    
## plot the results
## with no trimming


"""
turnovers=[big_fat_results_dict['turnover'][code_name][0] for code_name in parameter_set_names]


fig, ax = plt.subplots()
rects1 = ax.bar(np.arange(len(turnovers))+.5, turnovers, 0.5)
plt.xticks(np.arange(len(turnovers))+0.75, parameter_set_names, rotation="vertical")
plt.subplots_adjust(bottom=.4)

frame=plt.gca()


rcParams.update({'font.size': 18})


file_process("Basicturnover_%s_%dK_%s" % (country, int(start_portfolio_value/1000.0), datatype))

plt.show()

"""

turnover=pd.DataFrame([thing['turnover']['Baseline'] for thing in really_big_fat])
turnover.columns=windowsizes
turnover.index=min_sizes

turnover.transpose().plot()
plt.show()

costs=pd.DataFrame([thing['costs']['Baseline'] for thing in really_big_fat])
costs.columns=windowsizes
costs.index=min_sizes

costs.transpose().plot()
plt.show()

netreturn=pd.DataFrame([thing['grossreturn']['Baseline'] for thing in really_big_fat])
netreturn.columns=windowsizes
netreturn.index=min_sizes

netreturn.transpose().plot()
plt.show()


netreturn=pd.DataFrame([thing['netreturn']['Baseline'] for thing in really_big_fat])
netreturn.columns=windowsizes
netreturn.index=min_sizes

netreturn.transpose().plot()
plt.show()


netreturn=pd.DataFrame([thing['percentile_gross']['Baseline'] for thing in really_big_fat])
netreturn.columns=windowsizes
netreturn.index=min_sizes

netreturn.transpose().plot()
plt.show()

netreturn=pd.DataFrame([thing['percentile_net']['Baseline'] for thing in really_big_fat])
netreturn.columns=windowsizes
netreturn.index=min_sizes

netreturn.transpose().plot()
plt.show()


"""
turnover=pd.DataFrame([thing['turnover']['Baseline'] for thing in really_big_fat])
turnover.columns=windowsizes
turnover.index=min_sizes

turnover.transpose().plot(style=['b-','r--','k-.'])

plt.yticks([0.0, 0.3, 0.6])
plt.xticks([0.0, 0.5, 2.0])

rcParams.update({'font.size': 18})

file_process("Baselineturnover_%s_%dK_%s" % (country, int(start_portfolio_value/1000.0), datatype))

plt.show()
    
ax=plt.subplot(111)

costs=pd.DataFrame([thing['costs']['Baseline'] for thing in really_big_fat])*100.0
costs.columns=windowsizes
costs.index=min_sizes

costs.transpose().plot(style=['b-','r--','k:'])


plt.yticks([0.0, 2.0, 4.0])
plt.xticks([0.0, 0.5, 2.0])



rcParams.update({'font.size': 18})

file_process("Baselinecosts_%s_%dK_%s" % (country, int(start_portfolio_value/1000.0), datatype))

plt.show()

"""

"""
costs=[big_fat_results_dict['costs'][code_name][0]*100.0 for code_name in parameter_set_names]


fig, ax = plt.subplots()
rects1 = ax.bar(np.arange(len(costs))+.5, costs, 0.5)
plt.xticks(np.arange(len(costs))+0.75, parameter_set_names, rotation="vertical")
plt.subplots_adjust(bottom=.4)

frame=plt.gca()


rcParams.update({'font.size': 18})


file_process("Costs_%s_%dK_%s_%.2f_%.2f" % (country, int(start_portfolio_value/1000.0), datatype, min_trade_size, windowsize1))

plt.show()
"""