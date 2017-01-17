from optimisation import *
from matplotlib.pyplot import plot, show, xticks, xlabel, ylabel, legend, yscale, title, savefig, rcParams, figure, hist, text, bar, subplots
import matplotlib.pyplot as plt
import Image

from itertools import cycle

from datetime import datetime as dt
from twisted.test.test_amp import THING_I_DONT_UNDERSTAND



lines = ["-","--","-."]
linecycler = cycle(lines)    
colorcycler=cycle(["red", "blue", "green"])

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

rawdata=read_ts_csv("/home/rob/workspace/systematictradingexamples/plots_for_perhaps/MSCI_data.csv")

asset_returns2=calc_asset_returns(rawdata, ["JAPAN", "AUSTRALIA", "NEW_ZEALAND", "HONG_KONG", "SINGAPORE"])

corr=asset_returns2.corr().values()

def basic_opt(std,corr,mus):
    number_assets=mus.shape[0]
    sigma=sigma_from_corr(std, corr)
    start_weights=[1.0/number_assets]*number_assets
    
    ## Constraints - positive weights, adding to 1.0
    bounds=[(0.0,1.0)]*number_assets
    cdict=[{'type':'eq', 'fun':addem}]

    return minimize(neg_SR_riskfree, start_weights, (sigma, mus), method='SLSQP', bounds=bounds, constraints=cdict, tol=0.00001)

std=np.array([[.08]*asset_returns2.shape[1]]).transpose()
mus=np.array([[0.04]*asset_returns2.shape[1]]).transpose()
basic_opt(std, corr, mus)
