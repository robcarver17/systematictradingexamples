import pandas as pd
import datetime as dt
import Image
from random import gauss
import numpy as np
from matplotlib.pyplot import plot, show, xticks, xlabel, ylabel, legend, yscale, title, savefig, rcParams, figure, hist, scatter

import matplotlib.pylab as plt
from itertools import cycle
import pickle
import pandas as pd

lines = ["--","-","-."]
linecycler = cycle(lines)    


import Quandl


def get_quandl(code):
    QUANDLDICT=dict(GOLD="COM/WLD_GOLD", CORN="COM/PMAIZMT_USD", CRUDE_W="COM/WLD_CRUDE_WTI")
    authtoken='qWXuZcdwzwQ2GJQ88sNb'
    quandldef=QUANDLDICT[code]
    print quandldef
    data = Quandl.get(quandldef, authtoken=authtoken)

    data.columns=["value"]    
    return data.iloc[:,0]


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
    ans.index = pd.to_datetime(ans[date_index_name]).values

    del ans[date_index_name]

    ans.index.name = None

    return ans


def get_spot_price(instrument_code):
    datapath= "/home/rob/workspace3/pysystemtrade/sysdata/legacycsv/"
    filename = datapath+ instrument_code + "_carrydata.csv"
    instrcarrydata = pd_readcsv(filename)

    return instrcarrydata.PRICE

def get_adj_price(instrument_code):
    datapath= "/home/rob/workspace3/pysystemtrade/sysdata/legacycsv/"
    filename = datapath+ instrument_code + "_price.csv"
    instrprice = pd_readcsv(filename)
    instrprice.columns=["value"]

    return instrprice.iloc[:,0]


def get_interest():
    
    authtoken='qWXuZcdwzwQ2GJQ88sNb'
    quandldef='FRED/INTDSRUSM193N'
    print quandldef
    data = Quandl.get(quandldef, authtoken=authtoken)
    data.columns=["value"]
    return data.iloc[:,0]/1200.0



instrument_code="CORN"
start_date=dict(GOLD=pd.datetime(1975,1,1), CORN=pd.datetime(1982,04,30), CRUDE_W=pd.datetime(1987,12,31))[instrument_code]

data1=get_quandl(instrument_code)[start_date:]

perc_data1= (data1 - data1.shift(1))/data1

data2=get_adj_price(instrument_code).reindex(data1.index).ffill()

perc_data2= (data2 - data2.shift(1))/data2

irate=get_interest()
irate=irate.reindex(perc_data2.index,method="ffill")

perc_data2=perc_data2+irate

data1=perc_data1+1.0
data1=data1.cumprod()

data2=perc_data2+1.0
data2=data2.cumprod()


#data2 = data2 - (data2.irow(0) - data1.irow(0))
data3=data1-data2
data=pd.concat([data1, data2], axis=1).ffill()

data.columns=["Spot", "Future"]
data.plot(style=["g-", "b--"])
legend(loc="upper left")

frame=plt.gca()

#frame.get_yaxis().set_visible(False)
#frame.set_ylim([0,50000])



rcParams.update({'font.size': 18})



def file_process(filename):
    fig = plt.gcf()
    fig.set_size_inches(18.5,10.5)
    fig.savefig("/home/rob/%s.png" % filename,dpi=300)
    fig.savefig("/home/rob/%sLOWRES.png" % filename,dpi=50)
    
    Image.open("/home/rob/%s.png" % filename).convert('L').save("/home/rob/%s.jpg" % filename)
    Image.open("/home/rob/%sLOWRES.png" % filename).convert('L').save("/home/rob/%sLOWRES.jpg" % filename)


file_process("contango_%s" % instrument_code)

show()


data3.plot()
show()



