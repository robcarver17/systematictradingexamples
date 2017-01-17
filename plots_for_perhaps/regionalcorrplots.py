from optimisation import *
from matplotlib.pyplot import plot, show, xticks, xlabel, ylabel, legend, yscale, title, savefig, rcParams, figure, hist, text, bar, subplots
import matplotlib.pyplot as plt
import Image

from itertools import cycle

from datetime import datetime as dt

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

data=calc_asset_returns(rawdata, ["DEV_NORTHAM", "DEV_EUROPE", "DEV_ASIA"])

def linehist(x, color="blue", linestyle="-", lw=1, passed_axis=None):
    y,binEdges =np.histogram(x)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    
    if passed_axis is None:
        plot(bincenters,y,'-', color=color, linestyle=linestyle, lw=lw) 
    else:
        passed_axis.plot(bincenters,y,'-', color=color, linestyle=linestyle, lw=lw)


"""
Plot bootstrapped statistics

Awful code only works with 3 assets
"""

"""
corrs=[]

monte_carlo=1000
monte_length=len(data)

allcorrs=[]
for unused_index in range(monte_carlo):
    bs_idx=[int(random.uniform(0,1)*len(data)) for i in range(monte_length)]
    returns=data.iloc[bs_idx,:] 
    cm=returns.corr().values
    clist=[cm[0][1], cm[0][2],cm[1][2]]
    corrs.append(clist)
    allcorrs=allcorrs+clist
    
codes=data.columns
corrnames=["America / Europe", "America / Asia", "Europe / Asia", "All"]

corrs=np.array(corrs)

linehist(corrs[:,0], linestyle=next(linecycler), color=next(colorcycler))
linehist(corrs[:,1], linestyle=next(linecycler), color=next(colorcycler))
linehist(corrs[:,2], linestyle=next(linecycler), color=next(colorcycler))
linehist(allcorrs, lw=3, color="black")

allcorrs.sort()
point75=allcorrs[int(len(allcorrs)*.75)]
point50=np.median(allcorrs)
plt.axvline(point75, linestyle="--", color="black")
plt.axvline(point50, linestyle="--", color="black")

legend(corrnames, loc="upper left")

xlabel("Correlation")


frame=plt.gca()

frame.get_yaxis().set_visible(False)
#frame.set_xticks([ -0.0, 0.25, 0.5])



rcParams.update({'font.size': 18})


file_process("corrregionals")

show()


data=calc_asset_returns(rawdata, ["EM_EMEA", "EM_LATAM", "EM_ASIA"])

corrs=[]

monte_carlo=1000
monte_length=len(data)

allcorrs=[]
for unused_index in range(monte_carlo):
    bs_idx=[int(random.uniform(0,1)*len(data)) for i in range(monte_length)]
    returns=data.iloc[bs_idx,:] 
    cm=returns.corr().values
    clist=[cm[0][1], cm[0][2],cm[1][2]]
    corrs.append(clist)
    allcorrs=allcorrs+clist
    
codes=data.columns
corrnames=["EMEA / Latam", "EMEA / Asia", "Latam / Asia"]

corrs=np.array(corrs)

linehist(corrs[:,0], linestyle=next(linecycler), color=next(colorcycler))
linehist(corrs[:,1], linestyle=next(linecycler), color=next(colorcycler))
linehist(corrs[:,2], linestyle=next(linecycler), color=next(colorcycler))
linehist(allcorrs, lw=3, color="black")

allcorrs.sort()
point75=allcorrs[int(len(allcorrs)*.75)]
point50=np.median(allcorrs)
plt.axvline(point75, linestyle="--", color="black")
plt.axvline(point50, linestyle="--", color="black")

legend(corrnames, loc="upper left")

xlabel("Correlation")


frame=plt.gca()

frame.get_yaxis().set_visible(False)
#frame.set_xticks([ -0.0, 0.25, 0.5])



rcParams.update({'font.size': 18})

def file_process(filename):
    fig = plt.gcf()
    fig.set_size_inches(18.5,10.5)
    fig.savefig("/home/rob/%s.png" % filename,dpi=300)
    fig.savefig("/home/rob/%sLOWRES.png" % filename,dpi=50)
    
    Image.open("/home/rob/%s.png" % filename).convert('L').save("/home/rob/%s.jpg" % filename)
    Image.open("/home/rob/%sLOWRES.png" % filename).convert('L').save("/home/rob/%sLOWRES.jpg" % filename)

file_process("corrregionals")

show()


data=calc_asset_returns(rawdata, ["EM", "DEV"])
corrs=[]

monte_carlo=100
monte_length=len(data)

for unused_index in range(monte_carlo):
    bs_idx=[int(random.uniform(0,1)*len(data)) for i in range(monte_length)]
    returns=data.iloc[bs_idx,:] 
    cm=returns.corr().values
    corrs.append(cm[0][1])
    
corrs.sort()
point75=corrs[int(len(corrs)*.75)]



data=calc_asset_returns(rawdata, ["EM_EMEA", "EM_LATAM", "EM_ASIA", "DEV_EUROPE", "DEV_NORTHAM", "DEV_ASIA"])

corrstack=[]
corrs=[]

monte_carlo=100
monte_length=len(data)

for unused_index in range(monte_carlo):
    bs_idx=[int(random.uniform(0,1)*len(data)) for i in range(monte_length)]
    returns=data.iloc[bs_idx,:] 
    cm=returns.corr().values
    cm=np.tril(cm, k=-1)
    cm[cm==0]=np.NAN
    cx=[]
    for ci in cm:
        cx=cx+list(ci)
        
    clist=cx
    corrs.append(clist)

corrs=np.array(corrs)
corrs=corrs.transpose()

for citem in corrs:
    if not all(np.isnan(citem)):
        corrstack.append(list(citem))

allcorrs = sum(corrstack, [])

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()

linehist(allcorrs, lw=3, color="black", passed_axis=ax1)

    
for cs in corrstack:
    linehist(cs,  color="gray", passed_axis=ax2)


allcorrs.sort()
point75=allcorrs[int(len(allcorrs)*.75)]
point50=np.median(allcorrs)
plt.axvline(point75, linestyle="--", color="black")
plt.axvline(point50, linestyle="--", color="black")

show()
"""



regions=dict(emea_dev=["UK", "IRELAND", "GERMANY", "AUSTRIA", "SWITZERLAND", "NETHERLANDS", "BELGIUM", "SWEDEN", "FINLAND",
                      "NORWAY", "FRANCE", "ITALY", "SPAIN", "PORTUGAL", "ISRAEL"],
                asia_dev=["JAPAN", "AUSTRALIA", "NEW_ZEALAND", "HONG_KONG", "SINGAPORE"],
                america_dev=["USA", "CANADA"],
                latam_em=["BRAZIL", "MEXICO", "CHILE", "COLOMBIA", "PERU"],
               emea_em=[ "RUSSIA", "POLAND", "HUNGARY", "CZECH", "GREECE", "TURKEY", "QATAR",
                     "UAE", "EGYPT", "SOUTH_AFRICA"],
               asia_em=["CHINA", "INDIA", "TAIWAN", "KOREA", "MALAYSIA", "INDONESIA", "THAILAND", 
                     "PHILLIPINES"])

devlist=["emea_dev", "asia_dev", "america_dev"]
emlist=["latam_em", "emea_em", "asia_em"]

regionlist=emlist+devlist

corrstack=[]
for region in regionlist:
    codes=regions[region]
    data=calc_asset_returns(rawdata, codes)
    corrs=[]

    monte_carlo=100
    monte_length=len(data)
    
    for unused_index in range(monte_carlo):
        bs_idx=[int(random.uniform(0,1)*len(data)) for i in range(monte_length)]
        returns=data.iloc[bs_idx,:] 
        cm=returns.corr().values
        cm=np.tril(cm, k=-1)
        cm[cm==0]=np.NAN
        cx=[]
        for ci in cm:
            cx=cx+list(ci)
            
        clist=cx
        corrs.append(clist)
    
    corrs=np.array(corrs)
    corrs=corrs.transpose()
    
    for citem in corrs:
        if not all(np.isnan(citem)):
            corrstack.append(list(citem))


fig, ax1 = plt.subplots()

ax2 = ax1.twinx()



    
for cs in corrstack:
    linehist(cs,  color="lightskyblue", passed_axis=ax2)
linehist(corrstack[113],  color="black",  passed_axis=ax2)

allcorrs = sum(corrstack, [])

allcorrs.sort()
point75=allcorrs[int(len(allcorrs)*.75)]
point50=np.median(allcorrs)
plt.axvline(point75, linestyle="--", color="black")
plt.axvline(point50, linestyle="--", color="black")

linehist(allcorrs, lw=4, color="black", passed_axis=ax1)

frame=plt.gca()

ax1.get_yaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)
frame.set_xticks([ 0.0,  0.5])
ax2.set_ylim([0.0, 70.0])

rcParams.update({'font.size': 18})

file_process("intraregioncorr")

show()
