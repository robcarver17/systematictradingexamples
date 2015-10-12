import pandas as pd
import numpy as np
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
    
    stack_index=pd.date_range(start=start_date, periods=len(stack_values), freq="H")
    
    stacked=pd.TimeSeries(stack_values, index=stack_index)
    
    
    return stacked
     