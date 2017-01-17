"""
test the 12 month asset allocation model

no costs currently
"""

import pandas as pd
import numpy as np

def file_process(filename):
    fig = plt.gcf()
    fig.set_size_inches(18.5,10.5)
    fig.savefig("/home/rob/%s.png" % filename,dpi=300)
    fig.savefig("/home/rob/%sLOWRES.png" % filename,dpi=50)
    
    Image.open("/home/rob/%s.png" % filename).convert('L').save("/home/rob/%s.jpg" % filename)
    Image.open("/home/rob/%sLOWRES.png" % filename).convert('L').save("/home/rob/%sLOWRES.jpg" % filename)

data=pd.read_csv("/home/rob/workspace/systematictradingexamples/plots_for_perhaps/USmonthlyreturns.csv")
data.index=data.Date


SRdifftable=[     -0.25, -0.2, -0.15,  -0.1, -0.05, 0.0, 0.05, 0.1,  0.15, 0.2, 0.25]
multipliers=[.6,    .66, .77,     .85,  .94,   1.0, 1.11, 1.19, 1.3, 1.37, 1.48]

vols=[0.15, 0.08]

risk_weights=[.5, .5]


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


def calc_cash_weights(risk_weights):
    cash_weights=[rw/vl for (rw, vl) in zip(risk_weights, vols)]
    adj_wts=sum(cash_weights)/sum(risk_weights)
    cash_weights=[wt/adj_wts for wt in cash_weights]
    return cash_weights

def ts_calc_cash_weights(risk_weights_ts):
    return risk_weights_ts.apply(calc_cash_weights, 1)

def risk_weights_with_model(SRdiffs, risk_weights, SRdifftable, multipliers, absolute=False):
    
    multiplier_equity=np.interp(SRdiffs[0], SRdifftable, multipliers)
    multipliers_bonds=np.interp(-SRdiffs[1], SRdifftable, multipliers)
    
    new_risk_weights=[risk_weights[0]*multiplier_equity, risk_weights[1]*multipliers_bonds]
    
    if sum(new_risk_weights)>1.0 or not absolute:
        ## Normalise...
        new_risk_weights=[wt/sum(new_risk_weights) for wt in new_risk_weights]
    
    return new_risk_weights

def yield_forecast(data, vols):
    eqyield = data.SP_Yield/100.0
    bondyield = data.Bond_Yield/100.0
    
    eqreturn = eqyield / vols[0]
    bondreturn = bondyield / vols[1]

    avgreturn = pd.concat([eqreturn, bondreturn], axis=1).mean(axis=1)
    
    diff = eqreturn - avgreturn
    
    return diff*3.0

def yield_forecast_ts(data, vols):
    eqyield = data.SP_Yield
    bondyield = data.Bond_Yield
    
    eqreturn = eqyield / vols[0]
    bondreturn = bondyield / vols[1]

    eqreturn = eqreturn - pd.rolling_mean(eqreturn, 240, min_periods=1)
    bondreturn = bondreturn - pd.rolling_mean(bondreturn, 240, min_periods=1)


    diff = eqreturn - bondreturn
    
    return diff*3.0



def trailing_forecast( data, vols):
    
    eqreturn=(data.SP_TR_Index / data.SP_TR_Index.shift(12))-1
    bondreturn=(data.Bond_TR_Index / data.Bond_TR_Index.shift(12))-1
    
    eqreturn=eqreturn / vols[0]
    bondreturn=bondreturn / vols[1]
    
        
    return [eqreturn*.25, bondreturn*.25]

def calc_ts_new_risk_weights(data, vols, risk_weights, SRdifftable, multipliers, forecast_func, absolute=False):
    
    forecasts=forecast_func( data, vols)
    forecasts=pd.concat(forecasts,axis=1)
    new_weights=[risk_weights_with_model(list(forecasts.irow(frow).values), risk_weights, SRdifftable, multipliers, absolute) for 
                 frow in range(len(forecasts))]
    
    return pd.DataFrame(new_weights, data.Date, columns=["Equity", "Bond"])

def calc_ts_fixed_weights(data, risk_weights):
    return pd.DataFrame([risk_weights]*len(data.index), data.Date, columns=["Equity", "Bond"])
    
def portfolio_return(data, cash_weights):
    
    asset_returns=data[["SP_TR", "Bond_TR"]]
    asset_returns.columns=["Equity", "Bond"]
    asset_returns.index=data.Date
    
    portfolio_returns=asset_returns*cash_weights.shift(1)
    
    portfolio_returns=portfolio_returns.sum(axis=1)
    
    return portfolio_returns

def two_year_fix(weights):
    new_index=weights.index[range(0,len(weights.index), 60)]
    fix_weights=weights.reindex(new_index.unique())
    fix_weights=fix_weights.reindex(weights.index).ffill()

    return fix_weights

def analyse_returns(prets):
    basearithmean = prets.mean()*12
    new_std=prets.std()*(12**.5)
    variance=new_std**2
    gmm=basearithmean- - variance/2.0
    gsr=(gmm) / new_std

    return (basearithmean,  gmm, new_std, gsr)


p1=portfolio_return(data, ts_calc_cash_weights(calc_ts_fixed_weights(data, risk_weights)))
p2=portfolio_return(data, ts_calc_cash_weights(calc_ts_new_risk_weights(data, vols, risk_weights, SRdifftable, multipliers, trailing_forecast, absolute=False)))
p3=portfolio_return(data, ts_calc_cash_weights(calc_ts_new_risk_weights(data, vols, risk_weights, SRdifftable, multipliers, trailing_forecast, absolute=True)))

#p4=portfolio_return(data, two_year_fix(ts_calc_cash_weights(calc_ts_new_risk_weights(data, vols, risk_weights, SRdifftable, multipliers, yield_forecast))))
#p5=portfolio_return(data, two_year_fix(ts_calc_cash_weights(calc_ts_new_risk_weights(data, vols, risk_weights, SRdifftable, multipliers, yield_forecast_ts))))

#p1=p1/p1.std()
#p2=p2/p2.std()
#p3=p3/p3.std()
#p4=p4/p4.std()

p4=p3*p2.std()/p3.std()

analyse_returns(p1)
analyse_returns(p2)
analyse_returns(p3)
analyse_returns(p4)

p1=p1.cumsum()
p2=p2.cumsum()
p3=p3.cumsum()
p4=p4.cumsum()

thing=p4 - p3
thing.plot()
plt.show()


from matplotlib.pyplot import plot, show, xticks, xlabel, ylabel, legend, yscale, title, savefig, rcParams, figure, hist, text, bar, subplots
import matplotlib.pyplot as plt
import Image



eqyield = data.SP_Yield/100.0
bondyield = data.Bond_Yield/100.0
eqyield.plot()
bondyield.plot()
show()


eqreturn = eqyield / vols[0]
bondreturn = bondyield / vols[1]
eqreturn.index=[x[6:] for x in eqreturn.index]
bondreturn.index=[x[6:] for x in bondreturn.index]

eqreturn.plot(color="black")
bondreturn.plot(color="gray")
legend( ["Equities", "Bonds"], loc="upper left")
frame=plt.gca()

#frame.get_yaxis().set_visible(False)

rcParams.update({'font.size': 18})

file_process("yieldhistory")

plt.show()




diff = eqreturn - bondreturn

diff.plot()
show()


p1.index=[x[6:] for x in p1.index]
p2.index=[x[6:] for x in p2.index]
p3.index=[x[6:] for x in p3.index]
p4.index=[x[6:] for x in p4.index]


p1.plot(color="blue", linestyle="-.")
p2.plot(color="gray")
p3.plot(color="black", linestyle="--")
p4.plot(color="black")
legend( ["Fixed weight", "Relative momentum", "Absolute momentum", "Norm absolute momentum"], loc="upper left")
frame=plt.gca()

frame.get_yaxis().set_visible(False)

rcParams.update({'font.size': 18})

file_process("assetallocationmodel")

plt.show()




p1.plot(color="black")
p3.plot(color="gray")
legend( ["Fixed weight", "Model weight"], loc="upper left")
frame=plt.gca()

frame.get_yaxis().set_visible(False)

rcParams.update({'font.size': 18})

file_process("assetallocationmodel2")

plt.show()


from copy import copy
from datetime import datetime as dt

from scipy import stats

rawdata=copy(data)
tickers=["Stocks", "Bonds"]
rawdata.index=data.Date



def get_forward_tr(tickname, rawdata, months):
    asset_returns=data[["SP_TR", "Bond_TR"]]
    asset_returns.columns=tickers
    asset_returns.index=data.Date

    total_returns=asset_returns.cumsum()[tickname]
    return (total_returns.shift(-months) / total_returns) - 1.0


vol_ts=pd.DataFrame([vols]*len(rawdata.index), rawdata.index, columns=tickers)
vol_ts.columns=tickers

alldatatoplot_slope=dict(relative_div=[], relative_ts_div=[], momentum=[])
alldatatoplot_pvalue=dict(relative_div=[], relative_ts_div=[], momentum=[])

for nmonths in [1,3,6,12,24,36,60,120]:
    fwd_return=[get_forward_tr(tickname, rawdata, nmonths) for tickname in tickers]
    fwd_return=pd.concat(fwd_return, axis=1)
    fwd_return.columns=tickers
    
    fwd_SR=fwd_return / (vol_ts * ((nmonths/12.0)**.5))
    rel_SR=fwd_SR.Stocks - fwd_SR.Bonds
    
    stacked_SR=rel_SR.values
    
    predictors=[yield_forecast(data, vols).shift(1), 
                yield_forecast_ts(data, vols).shift(1),
                trailing_forecast(data, vols).shift(1)]
    
    for (predname, predset) in zip(["relative_div","relative_ts_div","momentum"], predictors):
        stack_pred=list(predset.values)
        valididx=[~np.isnan(stacked_SR[idx]) & ~np.isnan(stack_pred[idx]) for idx in range(len(stacked_SR))]
        stack_pred_valid=[stack_pred[idx] for idx in range(len(stack_pred)) if valididx[idx]]
        stack_SR_valid=[stacked_SR[idx] for idx in range(len(stacked_SR)) if valididx[idx]]
        
        slope, intercept, r_value, p_value, std_err=stats.linregress(stack_pred_valid, stack_SR_valid)
        
        print("%d months ahead, predictor %s, slope %.3f p_value %.4f R2 %.4f" % (
              nmonths, predname, slope, p_value, r_value))

        alldatatoplot_slope[predname].append(slope)
        alldatatoplot_pvalue[predname].append((1.0 - p_value)*sign(slope))

from matplotlib.pyplot import show, plot, legend, title

plot([1,3,6,12,24,36,60,120], alldatatoplot_slope["momentum"])
plot([1,3,6,12,24,36,60,120], alldatatoplot_slope["relative_div"])
plot([1,3,6,12,24,36,60,120], alldatatoplot_slope["relative_ts_div"])
legend(["momentum", "relative_div", "relative_ts_div"])

show()

plot([1,3,6,12,24,36,60,120], alldatatoplot_pvalue["momentum"])
plot([1,3,6,12,24,36,60,120], alldatatoplot_pvalue["relative_div"])
plot([1,3,6,12,24,36,60,120], alldatatoplot_pvalue["relative_ts_div"])
legend(["momentum", "relative_div", "relative_ts_div"])

show()
