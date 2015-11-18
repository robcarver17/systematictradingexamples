import pandas as pd
import numpy as np
import scipy.stats as st
DAYS_IN_YEAR=256.0
ROOT_DAYS_IN_YEAR=DAYS_IN_YEAR**.5

useroot=""

def cap_forecast(xrow, capmin,capmax):
    """
    Cap forecasts.
    
    """

    ## Assumes we have a single column    
    x=xrow[0]

    if x<capmin:
        return capmin
    elif x>capmax:
        return capmax
    
    return x

def cap_series(xseries, capmin=-20.0,capmax=20.0):
    """
    Apply capping to each element of a time series
    For a long only investor, replace -20.0 with 0.0

    """
    return xseries.apply(cap_forecast, axis=1, args=(capmin, capmax))

def get_list_code():
    ans=pd.read_csv("%sconfig.csv" % useroot)
    return list(ans.Instrument)

def get_point_sizes():
    ans=pd.read_csv("%sconfig.csv" % useroot)
    psizes=dict([(x[1].Instrument, float(x[1].Pointsize)) for x in ans.iterrows()])
    return psizes


def pd_readcsv(filename):
    """
    Reads the pandas dataframe from a filename, given the index is correctly labelled
    """

    
    ans=pd.read_csv(filename)
    
    ans.index=pd.to_datetime(ans['DATETIME'])
    del ans['DATETIME']
    ans.index.name=None
    return ans

def find_datediff(data_row):
    """
    data differential for a single row
    """
    if np.isnan(data_row.NEAR_MONTH) or np.isnan(data_row.TRADE_MONTH):
        return np.nan
    nearest_dt=pd.to_datetime(str(int(data_row.NEAR_MONTH)), format="%Y%m")
    trade_dt=pd.to_datetime(str(int(data_row.TRADE_MONTH)), format="%Y%m")
    
    distance = trade_dt - nearest_dt
    distance_years=distance.days/365.25
    
    ## if nearder contract is cheaper; price will fall
    price_diff=data_row.NEARER - data_row.TRADED 
    
    return price_diff/distance_years
    


def ewmac_forecast_scalar(Lfast, Lslow):
    """
    Function to return the forecast scalar (table 49 of the book)
    
    Only defined for certain values
    """
    
    
    fsdict=dict(l2_8=10.6, l4_16=7.5, l8_32=5.3, l16_64=3.75, l32_128=2.65, l64_256=1.87)
    
    lkey="l%d_%d" % (Lfast, Lslow)
    
    if lkey in fsdict:
        return fsdict[lkey]
    else:
        print "Warning: No scalar defined for Lfast=%d, Lslow=%d, using default of 1.0" % (Lfast, Lslow)
        return 1.0

def get_price_for_instrument(code):
    
    filename="%sdata/%s_price.csv" %  (useroot, code)
    price=pd_readcsv(filename)
    return price

def get_carry_data(code):
    
    filename="%sdata/%s_carrydata.csv" % (useroot, code)
    data=pd_readcsv(filename)

    return data

def uniquets(df3):
    """
    Makes x unique
    """
    df3=df3.groupby(level=0).first()
    return df3


def daily_resample(b, a):
    """
    Returns b dataframe resampled to a dataframe index
    
    """
    
    master_index=a.index
    a_daily=a.resample('1D') ## Only want index, fill method is irrelevant
    b=uniquets(b)
    b_daily=b.reindex(a_daily.index, method="ffill", limit=1)
    new_b=b_daily.reindex(master_index, method="ffill", limit=1)
    
    return new_b



def calculate_pandl(position_ts, price_ts, pointsize=1.0):
    rs_positions_ts=daily_resample(position_ts, price_ts).ffill()
    rets=price_ts - price_ts.shift(1)
    local_rets=rs_positions_ts.shift(1)*rets*pointsize

    return local_rets

def annualised_rets(total_rets):
    mean_rets=total_rets.mean(skipna=True)
    annualised_rets=mean_rets*DAYS_IN_YEAR
    return annualised_rets

def annualised_vol(total_rets):
    actual_total_daily_vol=total_rets.std(skipna=True)
    actual_total_annual_vol=actual_total_daily_vol*ROOT_DAYS_IN_YEAR
    return actual_total_annual_vol

def sharpe(total_rets):
    
    sharpe=annualised_rets(total_rets)/annualised_vol(total_rets)
    
    return sharpe

def stack_ts(tslist, start_date=pd.datetime(1970,1,1)):
    
    """
    Take a list of time series, and stack them, generating a new time series
    """
    

    tslist_values=[list(x.iloc[:,0].values) for x in tslist]
    stack_values=sum(tslist_values, [])
    stack_values=[x for x in stack_values if not np.isinf(x)]
    
    stacked=arbitrary_timeindex(stack_values, start_date)
    
    
    return stacked

def slices_for_ts(data, freq="12M"):
    """
    Return date indices for slicing up a data frame
    """
    yridx=list(pd.date_range(start=data.index[0], end=data.index[-1], freq=freq))
    yridx_stub=list(pd.date_range(start=yridx[-1], periods=2, freq=freq))[-1]
    yridx=yridx+[yridx_stub]
    
    return yridx

def break_up_ts(data, freq="12M"):
    """
    Take a data frame and break it into chunks 
    returns a list of data frames
    """
    yridx=slices_for_ts(data, freq)
    
    brokenup=[]
    for idx in range(len(yridx))[1:]:
        brokenup.append(data[yridx[idx-1]:yridx[idx]])
    
    return brokenup
    

def drawdown(x):
    ### Returns a ts of drawdowns for a time series x
    
    ## rolling max with infinite window
    maxx=pd.rolling_max(x, 99999999, min_periods=1)
    return (x - maxx)/maxx

    
    

class account_curve(pd.core.series.Series):
    """
    Inherits from pandas time series to give useful information
    
    Could be in % or GBP terms
    
    Downsamples to daily before doing anything else
    
    Can 
    
    """
    
    
    def new_freq(self, freq):
        ## Set up a new frequency.
        ## Note this will break certain things (eg Sharpe) so be careful
        if freq=="Daily":
            ## we assume we're daily so do nothing
            return self
        if freq=="Weekly":
            return self.cumsum().ffill().resample("W").diff()
        if freq=="Monthly":
            return self.cumsum().ffill().resample("M").diff()
    
    def sharpe(self):
        ## assumes daily returns
        return ROOT_DAYS_IN_YEAR*self.mean()/self.std()
    
    def annstd(self):
        return ROOT_DAYS_IN_YEAR*self.std()
    
    def losses(self):
        x=self.values
        return [z for z in x if z<0]
    
    def gains(self):
        x=self.values
        return [z for z in x if z>0]
    
    def avg_loss(self):
        return np.mean(self.losses())

    def avg_gain(self):
        return np.mean(self.gains())
    
    def drawdown(self):
        ## in case need numerous stats
        if "drawdownacc" not in dir(self):
            setattr(self, "drawdownacc", drawdown(cum_perc(self)))
        return self.drawdownacc
    
    def avg_drawdown(self):
        return self.perc_drawdown(50.0)
    
    def perc_drawdown(self, q):
        dd=self.drawdown()
        return np.percentile(dd, q)

    def worst_drawdown(self):
        dd=self.drawdown()
        return np.nanmin(dd.values)
        
    def time_in_drawdown(self):
        dd=self.drawdown()
        dd=[z for z in dd if not np.isnan(z)]
        in_dd=float(len([z for z in dd if z<0]))
        return in_dd/float(len(dd))
        
    def monthly_returns(self):
        return self.resample("1M", how="sum")
    
    def gaintolossratio(self):
        return self.avg_gain()/-self.avg_loss()
    
    def profitfactor(self):
        return sum(self.gains())/-sum(self.losses())
    
    def hitrate(self):
        no_gains=float(len(self.gains()))
        no_losses=float(len(self.losses()))
        return no_gains/(no_losses+no_gains)
    

def cum_perc(pd_timeseries):
    """
    Cumulate percentage returns for a pandas time series
    """
    
    cum_datalist=[1+x for x in pd_timeseries]
    cum_datalist=pd.TimeSeries(cum_datalist, index=pd_timeseries.index)
    
    
    return cum_datalist.cumprod()


def arbitrary_timeindex(Nperiods, index_start=pd.datetime(2000,1,1)):
    """
    For nice plotting, convert a list of prices or returns into an arbitrary pandas time series
    """    
    
    ans=pd.bdate_range(start=index_start, periods=Nperiods)
    
    return ans


def arbitrary_timeseries(datalist, index_start=pd.datetime(2000,1,1)):
    """
    For nice plotting, convert a list of prices or returns into an arbitrary pandas time series
    """    
    
    ans=pd.TimeSeries(datalist, index=arbitrary_timeindex(len(datalist), index_start))
    
    return ans

def remove_nans_from_list(xlist):
    return [x for x in xlist if not np.isnan(x)]



def autocorr(x, t=1):
    return np.corrcoef(np.array([x[0:len(x)-t], x[t:len(x)]]))[0,1]