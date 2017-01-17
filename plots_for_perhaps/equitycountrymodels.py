
from matplotlib.pyplot import plot, show, xticks, xlabel, ylabel, legend, yscale, title, savefig, rcParams, figure, hist, text, bar, subplots
import matplotlib.pyplot as plt
import Image

from scipy import stats
import pandas as pd
import numpy as np
from datetime import datetime as dt

def read_ts_csv(fname, dindex="Date"):
    data=pd.read_csv(fname)
    dateindex=[dt.strptime(dx, "%d/%m/%y") for dx in list(data[dindex])]
    data.index=dateindex
    del(data[dindex])
    
    return data


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
    
def get_monthly_tr(tickname, rawdata):
    total_returns=rawdata[tickname+"_TR"]
    return (total_returns / total_returns.shift(1)) - 1.0

def get_annual_tr(tickname, rawdata):
    total_returns=rawdata[tickname+"_TR"]
    return (total_returns / total_returns.shift(12)) - 1.0

def get_forward_tr(tickname, rawdata, months):
    total_returns=rawdata[tickname+"_TR"]
    return (total_returns.shift(-months) / total_returns) - 1.0


def risk_weight_with_model(SRdiff, old_weight):

    SRdifftable=[     -0.25, -0.2, -0.15,  -0.1, -0.05, 0.0, 0.05, 0.1,  0.15, 0.2, 0.25]
    multipliers=[.6,    .66, .77,     .85,  .94,   1.0, 1.11, 1.19, 1.3, 1.37, 1.48]
    
    """
    if SRdiff>0.25:
        SRdiff=0.25
    if SRdiff<-0.25:
        SRdiff=-0.25
    """
    multiplier=np.interp(SRdiff, SRdifftable, multipliers)
    
    new_weight=old_weight*multiplier

    return new_weight

def new_risk_weights(SRdiff_list, old_risk_weights):
    
    new_weights=[risk_weight_with_model(SRdiff, old_weight) 
                 for (SRdiff, old_weight) in zip(SRdiff_list, old_risk_weights)]
    
    total_weight=sum(new_weights)
    
    new_weights=[new_wt/total_weight for new_wt in new_weights]
    
    return new_weights
    
## 1/N

def calc_ts_weights(tickers, rawdata, SRdiff_list_ts):
    
    old_risk_weights=[1.0/len(tickers)]*len(tickers)
    
    new_weights=[new_risk_weights(SRdiff_list, old_risk_weights) for SRdiff_list
                 in SRdiff_list_ts]
    
    return pd.DataFrame(new_weights, rawdata.index, tickers)
    
def calc_ts_fixed_weights(rawdata, tickers, risk_weights):
    return pd.DataFrame([risk_weights]*len(rawdata.index), rawdata.index, columns=tickers)



def calc_cash_weights(risk_weights):
    cash_weights=[rw/vl for (rw, vl) in zip(risk_weights, vols)]
    cash_weights=[wt/sum(cash_weights) for wt in cash_weights]
    return cash_weights

def ts_calc_cash_weights(risk_weights_ts):
    return risk_weights_ts.apply(calc_cash_weights, 1)

def calc_asset_returns(rawdata, tickers):
    asset_returns=pd.concat([get_monthly_tr(tickname, rawdata) for tickname in tickers], axis=1)
    asset_returns.columns=tickers

    return asset_returns

def expand_boring(boring, tickers):
    return pd.DataFrame(np.array([boring.values]*len(tickers), ndmin=2).transpose(), boring.index, columns=tickers)

def two_year_fix(weights):
    fix_weights=weights.resample("1A")
    fix_weights=fix_weights.reindex(weights.index).ffill()

    return fix_weights

def relative_div_predictor(tickers, rawdata):
    all_divs=pd.concat([get_div_yield(tickname, rawdata) for tickname in tickers], axis=1, names=tickers)
    all_divs.columns=tickers
    
    vol_ts=pd.DataFrame([vols]*len(rawdata.index), rawdata.index, columns=tickers)

    divs_SR=all_divs / vol_ts

    SR_avg=divs_SR.median(axis=1)
    SR_avg=expand_boring(SR_avg, tickers)
    
    rel_SR=divs_SR - SR_avg
    
    return rel_SR

def relative_div(tickers, rawdata):

    rel_SR=relative_div_predictor(tickers, rawdata)
    
    SRdiff_list_ts=[list(rel_SR.irow(idx).values*2.5) for idx in range(len(rawdata.index))] 

    risk_weights_ts=calc_ts_weights(tickers, rawdata, SRdiff_list_ts)

    return risk_weights_ts

def relative_ts_div_predictor(tickers, rawdata):
    all_divs=pd.concat([get_div_yield(tickname, rawdata) for tickname in tickers], axis=1, names=tickers)
    all_divs.columns=tickers
   
    vol_ts=pd.DataFrame([vols]*len(rawdata.index), rawdata.index, columns=tickers)

    divs_SR=all_divs / vol_ts

    divs_SR_ts_avg=pd.rolling_mean(divs_SR, 600, min_periods=1)
    
    norm_SR=divs_SR - divs_SR_ts_avg

    SR_avg=norm_SR.median(axis=1)
    SR_avg=expand_boring(SR_avg, tickers)

    
    rel_SR=norm_SR - SR_avg

    return rel_SR

def relative_ts_div(tickers, rawdata):
    rel_SR = relative_ts_div_predictor(tickers, rawdata)
    SRdiff_list_ts=[list(rel_SR.irow(idx).values*2.5) for idx in range(len(rawdata.index))] 

    risk_weights_ts=calc_ts_weights(tickers, rawdata, SRdiff_list_ts)

    return two_year_fix(risk_weights_ts)

def mom_predictor(tickers, rawdata):
    all_mom=[get_annual_tr(tickname, rawdata) for tickname in tickers]
    all_mom=pd.concat(all_mom, axis=1)
    all_mom.columns=tickers

    vol_ts=pd.DataFrame([vols]*len(rawdata.index), rawdata.index, columns=tickers)

    mom_SR=all_mom / vol_ts

    SR_avg=mom_SR.median(axis=1)
    SR_avg=expand_boring(SR_avg, tickers)

    
    rel_SR=mom_SR - SR_avg

    return rel_SR


def momentum(tickers, rawdata):
    rel_SR=mom_predictor(tickers, rawdata)
    SRdiff_list_ts=[list(rel_SR.irow(idx).values*.5) for idx in range(len(rawdata.index))] 

    risk_weights_ts=calc_ts_weights(tickers, rawdata, SRdiff_list_ts)
    
    return risk_weights_ts

def mom_rel(tickers, rawdata):
    a1=momentum(tickers, rawdata)
    a2=relative_ts_div(tickers, rawdata)    
    
    return (a1+a2)/2.0

def fixed(tickers, rawdata):
    
    SRdiff_list_ts=[[0.0]*len(tickers)]*len(rawdata.index) 

    risk_weights_ts=calc_ts_weights(tickers, rawdata, SRdiff_list_ts)

    return risk_weights_ts


def portfolio_return(asset_returns, cash_weights):
    
    portfolio_returns=asset_returns*cash_weights
    
    portfolio_returns=portfolio_returns.sum(axis=1)
    
    return portfolio_returns

def port_return(tickers, rawdata, risk_wt_function):

    risk_weights_ts=risk_wt_function(tickers, rawdata)

    asset_returns=calc_asset_returns(rawdata, tickers)
    cash_weights=ts_calc_cash_weights(risk_weights_ts)
    cash_weights=~np.isnan(asset_returns)*cash_weights

    def normalise_weights(new_weights):
        total_weight=sum(new_weights) 
        return [new_wt/total_weight for new_wt in new_weights]
    
    cash_weights=cash_weights.apply(normalise_weights, axis=1)
    cash_weights=cash_weights.shift(1)
    
    ans=portfolio_return(asset_returns, cash_weights)
    return ans

def sign(x):
    if x is None:
        return None
    if np.isnan(x):
        return np.NAN
    if x==0:
        return 0.0
    if x<0:
        return -1.0
    if x>0:
        return 1.0

from matplotlib.pyplot import show, plot, legend, title

rawdata=read_ts_csv("/home/rob/workspace/systematictradingexamples/plots_for_perhaps/MSCI_data.csv")
refdata=pd.read_csv("/home/rob/workspace/systematictradingexamples/plots_for_perhaps/MSCI_ref.csv")


## Comparisions:
## Type=Country, etc

caselist=["emdev", "devregions", "emregions", "devemea", "devams", "devasia", "emasia", "emams",
          "ememea", "devall", "emall", "all", "emea", "america", "asia"]

caselist=["all"]

"""    
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
"""

tickers=list(refdata[refdata.Type=="Country"].Country.values) #mom 17bp


def vl(emordev):
    if emordev=="EM":
        return .23
    else:
        return .15
    
vols=[vl(refdata[refdata.Country==ticker].EmorDEV.values[0]) for ticker in tickers]

## Use a filter function for each comparision
## Returns a list of "tickers"

## replace this function

weights=pd.read_csv("/home/rob/workspace/systematictradingexamples/plots_for_perhaps/allhcweights.csv")

p1=port_return(tickers, rawdata, fixed)

p2=port_return(tickers, rawdata, relative_div)


#p3=port_return(tickers, rawdata, relative_ts_div)

p4=port_return(tickers, rawdata, momentum)

p5=port_return(tickers, rawdata, mom_rel)


"""
mom_predictor(tickers, rawdata).plot()
title("mom")
show()
"""

all=pd.concat([p1,p2,p4], axis=1)
#all.columns=[ "Fixed","Relative div", "Norm div", "mom", "mom+ndiv"]


def sharpe(x):
    return (12*.5)*x.mean()/x.std()

    
print(all.apply(sharpe, axis=0))

def file_process(filename):
    fig = plt.gcf()
    fig.set_size_inches(18.5,10.5)
    fig.savefig("/home/rob/%s.png" % filename,dpi=300)
    fig.savefig("/home/rob/%sLOWRES.png" % filename,dpi=50)
    
    Image.open("/home/rob/%s.png" % filename).convert('L').save("/home/rob/%s.jpg" % filename)
    Image.open("/home/rob/%sLOWRES.png" % filename).convert('L').save("/home/rob/%sLOWRES.jpg" % filename)

p1.cumsum().plot(color="blue", linestyle="-.")
p2.cumsum().plot(color="gray")
p4.cumsum().plot(color="black")
legend( ["Fixed weight", "Dividend yield", "Relative momentum"], loc="upper left")
frame=plt.gca()

frame.get_yaxis().set_visible(False)

rcParams.update({'font.size': 18})

file_process("equitycountrymodel")

plt.show()
