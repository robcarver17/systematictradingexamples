from optimisation import *
from matplotlib.pyplot import plot, show, xticks, xlabel, ylabel, legend, yscale, title, savefig, rcParams, figure, hist, text, bar, subplots
import matplotlib.pyplot as plt
import Image

from itertools import cycle

from datetime import datetime as dt
from twisted.test.test_amp import THING_I_DONT_UNDERSTAND



lines = ["-","--"]
linecycler = cycle(lines)    
colorcycler=cycle(["red", "blue"])

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

def pd_readcsv(filename, date_index_name="DATETIME"):
    """
    Reads a pandas data frame, with time index labelled
    package_name(/path1/path2.., filename

    :param filename: Filename with extension
    :type filename: str

    :param date_index_name: Column name of date index
    :type date_index_name: list of str


    :returns: pd.DataFrame

    """

    ans = pd.read_csv(filename)
    #ans.index = pd.to_datetime(ans[date_index_name]).values
    ans.index = ans[date_index_name].values

    del ans[date_index_name]

    ans.index.name = None

    return ans


data=pd_readcsv("/home/rob/workspace/systematictradingexamples/plots_for_perhaps/USrealreturns.csv", "year")
data=data[['SP500','US10']]

def linehist(x, color="blue", linestyle="-"):
    y,binEdges =np.histogram(x, bins=50)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    plot(bincenters,y,'-', color=color, linestyle=linestyle) 


"""
Plot bootstrapped statistics

Awful code only works with 3 assets
"""

def finalpr(x):
    prod_returns=x+1.0
    cum_prod_returns=prod_returns.cumprod()
    final_pr=cum_prod_returns.values[-1]

    return final_pr

def geomean(x):
    if type(x) is pd.core.frame.DataFrame:
        return x.apply(geomean, 0)

    final_pr=finalpr(x)
    avg_pr=final_pr**(1/float(len(x.index)))
    return avg_pr-1.0


srs=[]
stds=[]
corrs=[]
gmeans=[]
gdiff=[]
gdiffsr=[]

monte_carlo=10000

monte_length=len(data)

for unused_index in range(monte_carlo):
    bs_idx=[int(random.uniform(0,1)*len(data)) for i in range(monte_length)]
    returns=data.iloc[bs_idx,:]
    ## GEOMETRIC 
    gm=geomean(returns)
    gmeans.append(list(gm))
    gdiff.append(gm[0] - gm[1])
    st=returns.std()
    stds.append(list(st))
    sr=gm/st
    srs.append(list(sr))
    gdiffsr.append(sr[0] - sr[1])
    cm=returns.corr().values
    corrs.append([cm[0][1]])
    
codes=data.columns


gmeans=np.array(gmeans)
stds=np.array(stds)
corrs=np.array(corrs)
srs=np.array(srs)
gdiff=np.array(gdiff)
gdiffsr=np.array(gdiffsr)


linehist(gmeans[:,0], linestyle=next(linecycler), color=next(colorcycler))
linehist(gmeans[:,1], linestyle=next(linecycler), color=next(colorcycler))


frame=plt.gca()

frame.get_yaxis().set_visible(False)
frame.set_xticks([0.0, 0.10, 0.2])

legend(codes)

xlabel("Annualised geometric return")

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
legend(codes)

frame=plt.gca()

frame.get_yaxis().set_visible(False)
frame.set_xticks([ 0.0, 0.1,  0.2, 0.3])


print np.percentile(stds[:,0], 5)
print np.percentile(stds[:,0], 95)
print np.percentile(stds[:,1], 5)
print np.percentile(stds[:,1], 95)


xlabel("Annualised standard deviation")
rcParams.update({'font.size': 18})

file_process("stdhist")

show()


linehist(corrs[:,0])


xlabel("Correlation")


frame=plt.gca()

frame.get_yaxis().set_visible(False)
frame.set_xticks([ -0.25, 0, 0.25])

print np.percentile(corrs[:,0], 5)
print np.percentile(corrs[:,0], 95)


rcParams.update({'font.size': 18})

file_process("corrdist")

show()

linehist(srs[:,0], linestyle=next(linecycler), color=next(colorcycler))
linehist(srs[:,1], linestyle=next(linecycler), color=next(colorcycler))
legend(codes)

frame=plt.gca()

frame.get_yaxis().set_visible(False)
frame.set_xticks([0.0, 0.5])


xlabel("Sharpe Ratio")

rcParams.update({'font.size': 18})

file_process("SRdist")

show()



linehist(list(gdiff))


xlabel("Difference in geometric means")


frame=plt.gca()

frame.get_yaxis().set_visible(False)
frame.set_xticks([  0.0,  0.1])



rcParams.update({'font.size': 18})

file_process("gdiffdist")

show()




linehist(list(gdiffsr))


xlabel("Difference in Sharpe Ratios")


frame=plt.gca()

frame.get_yaxis().set_visible(False)
frame.set_xticks([ -.5, 0.0, .5])



rcParams.update({'font.size': 18})

file_process("diffsrdist")

print np.percentile(list(gdiffsr), 5)
print np.percentile(list(gdiffsr), 95)

show()

