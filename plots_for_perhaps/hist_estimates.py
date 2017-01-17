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

data=pd.read_csv("/home/rob/workspace/systematictradingexamples/plots_for_perhaps/USmonthlyreturns.csv")
data = data[1188:]
dindex="Date"
dateindex=[dt.strptime(dx, "%d/%m/%Y") for dx in list(data[dindex])]
data.index=dateindex
del(data[dindex])


asset_returns2=calc_asset_returns(rawdata, ["UK"])
asset_returns2.columns=["UK_Equity"]


asset_returns=data[["SP_TR", "Bond_TR"]]
asset_returns.columns=["US_Equity", "US_Bond"]

asset_returns.index = [x + pd.DateOffset(months=1) for x in asset_returns.index]
asset_returns.index = [x - pd.DateOffset(days=1) for x in asset_returns.index]


data=pd.concat([asset_returns, asset_returns2], axis=1)
data = data.cumsum().ffill().reindex(asset_returns.index).diff()

data.columns=["US Equity" ,"US Bond", "UK Equity"]

def linehist(x, color="blue", linestyle="-"):
    y,binEdges =np.histogram(x)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    plot(bincenters,y,'-', color=color, linestyle=linestyle) 


"""
Plot bootstrapped statistics

Awful code only works with 3 assets
"""


srs=[]
means=[]
stds=[]
corrs=[]

monte_carlo=100
monte_length=len(data)

for unused_index in range(monte_carlo):
    bs_idx=[int(random.uniform(0,1)*len(data)) for i in range(monte_length)]
    returns=data.iloc[bs_idx,:] 
    means.append(list(12.0*returns.mean()))
    stds.append(list((12**.5)*returns.std()))
    srs.append(list((12**.5)*returns.mean()/returns.std()))
    cm=returns.corr().values
    corrs.append([cm[0][1], cm[0][2],cm[1][2]])
    
codes=data.columns
corrnames=["US Equity / US Bond", "US Equity / UK Equity", "US Bond / UK Equity"]

means=np.array(means)
stds=np.array(stds)
corrs=np.array(corrs)
srs=np.array(srs)

linehist(means[:,0], linestyle=next(linecycler), color=next(colorcycler))
linehist(means[:,1], linestyle=next(linecycler), color=next(colorcycler))
linehist(means[:,2], linestyle=next(linecycler), color=next(colorcycler))

frame=plt.gca()

frame.get_yaxis().set_visible(False)
frame.set_xticks([0.0, 0.10, 0.2])

legend(codes)

xlabel("Annualised return")

rcParams.update({'font.size': 18})


def file_process(filename):
    fig = plt.gcf()
    fig.set_size_inches(18.5,10.5)
    fig.savefig("/home/rob/%s.png" % filename,dpi=300)
    fig.savefig("/home/rob/%sLOWRES.png" % filename,dpi=50)
    
    Image.open("/home/rob/%s.png" % filename).convert('L').save("/home/rob/%s.jpg" % filename)
    Image.open("/home/rob/%sLOWRES.png" % filename).convert('L').save("/home/rob/%sLOWRES.jpg" % filename)

file_process("meanhist")


show()


linehist(stds[:,0], linestyle=next(linecycler), color=next(colorcycler))
linehist(stds[:,1], linestyle=next(linecycler), color=next(colorcycler))
linehist(stds[:,2], linestyle=next(linecycler), color=next(colorcycler))
legend(codes)

frame=plt.gca()

frame.get_yaxis().set_visible(False)
frame.set_xticks([ 0.0, 0.1,  0.2, 0.3])



xlabel("Annualised standard deviation")
rcParams.update({'font.size': 18})

file_process("stdhist")

show()

linehist(corrs[:,0], linestyle=next(linecycler), color=next(colorcycler))
linehist(corrs[:,1], linestyle=next(linecycler), color=next(colorcycler))
linehist(corrs[:,2], linestyle=next(linecycler), color=next(colorcycler))
legend(corrnames, loc="upper center")

xlabel("Correlation")


frame=plt.gca()

frame.get_yaxis().set_visible(False)
frame.set_xticks([ -0.0, 0.25, 0.5])



rcParams.update({'font.size': 18})

file_process("corrdist")

show()

linehist(srs[:,0], linestyle=next(linecycler), color=next(colorcycler))
linehist(srs[:,1], linestyle=next(linecycler), color=next(colorcycler))
linehist(srs[:,2], linestyle=next(linecycler), color=next(colorcycler))
legend(codes)

frame=plt.gca()

frame.get_yaxis().set_visible(False)
frame.set_xticks([0.0, 0.5, 1.0])


xlabel("Sharpe Ratio")

rcParams.update({'font.size': 18})

file_process("SRdist")

show()


## relative_div

def relative_item(data, func=None, usecorr=False):

    if usecorr:
        mean_ts = mycorr(data)
    else:
        mean_ts=pd.rolling_apply(data, 60, func)
    
    avg_ts=mean_ts.mean(axis=1)
    avg_ts=pd.concat([avg_ts, avg_ts, avg_ts], axis=1)
    avg_ts.columns = mean_ts.columns
    
    if usecorr:
        mean_ts=mean_ts - avg_ts
    else:
        mean_ts=mean_ts / avg_ts
    
    return mean_ts

def sr(xdata):
    return (12**.5)*np.mean(xdata)/np.std(xdata)

def mycorr(data):
    c1=pd.rolling_corr(data['US Equity'], data['US Bond'], window=60)
    c2=pd.rolling_corr(data['US Equity'], data['UK Equity'], window=60)
    c3=pd.rolling_corr(data['US Bond'], data['UK Equity'], window=60)
    
    thing = pd.concat([c1,c2,c3], axis=1)
    thing.columns=["US Equity / US Bond", "US Equity / UK Equity", "US Bond / UK Equity"]
    
    return thing

mean_ts=relative_item(data, np.mean)
mean_ts.plot()
show()

std_ts=relative_item(data, np.std)
std_ts.plot()
show()

sr_ts = relative_item(data, sr)
sr_ts.plot()
show()

corr_ts=relative_item(data,  usecorr=True)
corr_ts.plot()

#frame=plt.gca()
#frame.set_ylim([-1.0, 1.0])
#frame.set_yticks([-1.0, 0.0, 1.0])

show()

