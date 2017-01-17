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


"""
compare:

handcrafting

bootstrapped

one shot

equal weights

market cap weights
"""

import pandas as pd
from datetime import datetime as dt

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

def portfolio_return(asset_returns, cash_weights):
    
    index_returns=asset_returns.cumsum().ffill().diff()
    
    cash_align = cash_weights.reindex(asset_returns.index, method="ffill")
    cash_align[np.isnan(index_returns)]=0.0
    cash_align[np.isnan(cash_align)]=0.0
    
    vols=pd.ewmstd(asset_returns, span=100, min_periods=1)
    riskweights=pd.DataFrame(cash_align.values / vols.values, index=vols.index)
    riskweights.columns=asset_returns.columns
    
    riskweights[np.isnan(riskweights)]=0.0
    
    def _rowfix(x):
        if all([y==0.0 for y in x]):
            return x
        sumx=sum(x)
        return [y/sumx for y in x]

    riskweights = riskweights.apply(_rowfix, axis=1)
    
    portfolio_returns=asset_returns*riskweights
    
    portfolio_returns[np.isnan(portfolio_returns)]=0.0
    
    portfolio_returns=portfolio_returns.sum(axis=1)
    
    return portfolio_returns

import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import numpy as np
from datetime import datetime as dt
import datetime 

from scipy.optimize import minimize
from copy import copy
import random

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
    
    
def handcraft_equal(returns):
    """
    dynamic handcrafting, equal weights only
    """

    ## RETURNS Correlation matrix
    use_returns=equalise_vols(returns, default_vol=16.0)
    

    ## Sigma matrix = correlations
    sigma=use_returns.cov()
    sigma[sigma<0.0]=0.0

    ungroupedreturns=dict([(x,returns[x]) for x in returns.columns])
    tree_data=hc_sigma(sigma, ungroupedreturns)
    
    tree_data=grouping_tree(tree_data)
    weights=tree_to_weights(tree_data)
    
    return weights

def hc_sigma(ungrouped_sigma, ungroupedreturns,  groupdata=None):
    """
    handcraft weights from sigma matrix
    
    Algo:
    - Find pair of assets with highest correlation
    - Form them into a new group with equal weights
    - The group becomes like a new asset
    - Once we only have two assets left, stop.
    
    Need to 
    """

    if len(ungroupedreturns)==1:
        return groupdata[1]
    
    if groupdata is None:
        ## first run
        ## groupdata stores grouping information
        ## To begin with each group just consists of one asset
        groupdata=[[],list(ungrouped_sigma.columns)]
        
    groupedreturns=dict()
    
    
    ## iteration
    while len(ungroupedreturns)>0:
        ## current_sigma consists of the correlation of things we currently have
        if len(ungroupedreturns)==1:
            idx_list=[0]
        else:
            idx_list=find_highest_corr(ungrouped_sigma)
    
        
        name_list=tuple([ungrouped_sigma.columns[idx] for idx in idx_list])
        
        ## pair those things up
        (ungrouped_sigma, ungroupedreturns, groupedreturns, 
         groupdata)=group_assets(ungrouped_sigma, ungroupedreturns, groupedreturns, groupdata, idx_list, name_list)
    
    new_returns=pd.concat(groupedreturns, axis=1)
    new_sigma=new_returns.corr()
    ## recursive
    return hc_sigma(new_sigma, groupedreturns,  groupdata=[[],groupdata[0]])
        


def find_highest_corr(sigmat):
    new_sigmat=copy(sigmat.values)
    np.fill_diagonal(new_sigmat, -100.0)
    (i,j)=np.unravel_index(new_sigmat.argmax(), new_sigmat.shape)
    return (i,j)

def group_assets(ungrouped_sigma, ungroupedreturns, groupedreturns, groupdata, idx_list, name_list):
    """
    Group assets
    """
    todelete=[]
    names=[]
    grouping=[]
    group_returns=[]
    
    weights=[1.0/len(idx_list)]*len(idx_list) ## could have more complex thing here...
    
    for (itemweight,idx, iname) in zip(weights,idx_list, name_list):
        gi=groupdata[1][idx]
        grouping.append(gi)
        gri=ungroupedreturns.pop(iname)
        
        group_returns.append(gri*itemweight)
        names.append(gri.name)
    
        ungrouped_sigma=ungrouped_sigma.drop(iname, axis=0)
        ungrouped_sigma=ungrouped_sigma.drop(iname, axis=1)
    
        todelete.append(idx)

    groupdata[0].append(grouping)
    gr_returns=pd.concat(group_returns, axis=1)
    gr_returns=gr_returns.sum(axis=1)
        
    gr_returns.name="[%s]" % "+".join(names)
        
    print "Pairing %s" % ", ".join(names)
    
    groupedreturns[gr_returns.name]=gr_returns
    groupdata[1]=[element for eindex, element in enumerate(groupdata[1]) if eindex not in todelete]

    return (ungrouped_sigma, ungroupedreturns, groupedreturns, 
         groupdata)


def grouping_tree(tree_data, sigma):
    """
    Group branches of 2 into larger if possible
    """
    pass

def corrs_in_group(group, sigma):
    asset_list=sum(group, [])
    littlesigma=sigma.loc[asset_list, asset_list]

def corr_from_leaf(leaf, sigma):
    return sigma[leaf[0]][leaf[1]]

def tree_to_weights(tree_data):
    """
    convert a tree into weights
    """
    pass

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
                          monte_carlo=100, monte_length=None, shrinkage_factors=(0.5, 0.5),
                          weightdf=None):
    """
    Do an optimisation
    
    Returns data frame of weights
    
    Note if fitting in sample weights will be somewhat boring
    
    Doesn't deal with eg missing data in certain subperiods
    
    
    """
    if monte_length is None:
        monte_length=int(len(data.index)*.1)
    
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
        
        elif fit_method=="fixed":
            weights=[float(weightdf[weightdf.Country==ticker].Weight.values) for ticker in list(period_subset_data.columns)]
        else:
            raise Exception("Fitting method %s unknown" % fit_method)
        
        ## We adjust dates slightly to ensure no overlaps
        dindex=[fit_tuple[2]+datetime.timedelta(seconds=1), fit_tuple[3]-datetime.timedelta(seconds=1)]
        
        ## create a double row to delineate start and end of test period
        weight_row=pd.DataFrame([weights]*2, index=dindex, columns=data.columns)
        
        weight_list.append(weight_row)
        
    weight_df=pd.concat(weight_list, axis=0)
    
    return weight_df




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



rawdata=read_ts_csv("/home/rob/workspace/systematictradingexamples/plots_for_perhaps/MSCI_data.csv")
refdata=pd.read_csv("/home/rob/workspace/systematictradingexamples/plots_for_perhaps/MSCI_ref.csv")
tickers=list(refdata[(refdata.EmorDEV=="DEV") & (refdata.Type=="Country")].Country.values) #mom 12bp
#tickers=list(refdata[refdata.Type=="Country"].Country.values) #mom 12bp

fix_hcweights=pd.read_csv("/home/rob/workspace/systematictradingexamples/plots_for_perhaps/devhcweights.csv")
fix_capweights=pd.read_csv("/home/rob/workspace/systematictradingexamples/plots_for_perhaps/devcapweights.csv")
fix_eqweights=pd.DataFrame(dict(Country=tickers, Weight=[1.0/len(tickers)]*len(tickers)))



data=calc_asset_returns(rawdata, tickers)

### IDEA: to boostrap the results
### Repeatedly draw from 'data' to make new pseudo series

oneperiodweights=optimise_over_periods(data, "expanding", "one_period", equalisemeans=False, equalisevols=True)
#bootstrapweights=optimise_over_periods(data, "expanding", "bootstrap", equalisemeans=True, equalisevols=True)
exposthcweights=optimise_over_periods(data, "expanding", "fixed", weightdf=fix_hcweights, equalisemeans=True, equalisevols=True)
equalweights=optimise_over_periods(data, "expanding", "fixed", weightdf=fix_eqweights, equalisemeans=True, equalisevols=True)
marketcapweights=optimise_over_periods(data, "expanding", "fixed", weightdf=fix_capweights, equalisemeans=True, equalisevols=True)

index_returns=(1.0+data).cumprod().ffill()
last_return=index_returns.irow(-1).values
last_return=pd.DataFrame(np.array([last_return]*len(data)), data.index)
last_return.columns=data.columns
index_returns = index_returns / last_return

marketcapweights = marketcapweights.reindex(index_returns.index, method="ffill")
marketcapweights=marketcapweights*index_returns

marketcapweights=marketcapweights.ffill()

## portfolio, take out missing weights
p1=portfolio_return(data, oneperiodweights)[pd.datetime(1994,1,1):]
#p2=portfolio_return(data, bootstrapweights)
p3=portfolio_return(data, exposthcweights)[pd.datetime(1994,1,1):]
p4=portfolio_return(data, equalweights)[pd.datetime(1994,1,1):]
p5=portfolio_return(data, marketcapweights)[pd.datetime(1994,1,1):]

drag1=p3 - p1
drag2=p4 - p5 

def stats(x):
    ann_mean=x.mean()*12
    ann_std = x.std()*(12**.5)
    geo_mean = ann_mean - (ann_std**2)/2.0
    sharpe = geo_mean / ann_std
    
    return (ann_mean, ann_std, geo_mean, sharpe)

print stats(p1)
print stats(p3)
print stats(p4)
print stats(p5)

toplot=pd.concat([p1, p3, p4, p5], axis=1)
toplot.columns=["Optimised", "Handcraft", "Equal", "Market Cap"]
toplot.cumsum().plot()

show()


p1.cumsum().plot(color="black", ls="solid")
p3.cumsum().plot(color="gray", ls="solid")
p4.cumsum().plot(color="black", ls="dashed")
p5.cumsum().plot(color="gray", ls="dashed")

legend( ["Optimised", "Handcraft", "Equal", "Market Cap"], loc="upper left")
frame=plt.gca()

#frame.get_yaxis().set_visible(False)

rcParams.update({'font.size': 18})

file_process("compareoptmethods")
show()


drag1.cumsum().plot(color="gray", ls="solid")
 
legend( [ "Handcraft vs MktCap"], loc="upper left")
frame=plt.gca()

#frame.get_yaxis().set_visible(False)

rcParams.update({'font.size': 18})

file_process("compareoptmethodstracking")
show()
