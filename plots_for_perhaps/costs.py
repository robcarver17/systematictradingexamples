
from optimisation import *
from matplotlib.pyplot import plot, show, xticks, xlabel, ylabel, legend, yscale, title, savefig, rcParams, figure, hist, text, bar, subplots
import matplotlib.pyplot as plt
import Image

from itertools import cycle

from datetime import datetime as dt

lines = ["-","--","-.", ]
linecycler = cycle(lines)    
colorcycler=cycle(["red", "blue"])

from random import gauss
import numpy as np
import pandas as pd

mean=0.05
costladder=[0.0, 0.0001, 0.001, 0.005, 0.01]

nlength=2016 - 1249

results=[]
for cost in costladder:
    x=[1+(mean - cost)]*nlength
    x=np.cumprod(x)
    
    x=x*100.0
    x=pd.Series(x, pd.date_range(pd.datetime(2016,1,1), periods=nlength, freq="D"))
    
    results.append(x)
    
results=pd.concat(results, axis=1)
rel_labels=["%d bp" % (cost*10000.0) for cost in costladder]
results.columns = rel_labels

for costlabel in rel_labels:
    results[costlabel].plot(linestyle=next(linecycler), color=next(colorcycler))
legend(loc="upper left")

frame=plt.gca()

#frame.get_xaxis().set_visible(False)
frame.set_yticks([0.0, 1000.0, 2000.0])




rcParams.update({'font.size': 18})

def file_process(filename):
    fig = plt.gcf()
    fig.set_size_inches(18.5,10.5)
    fig.savefig("/home/rob/%s.png" % filename,dpi=300)
    fig.savefig("/home/rob/%sLOWRES.png" % filename,dpi=50)
    
    Image.open("/home/rob/%s.png" % filename).convert('L').save("/home/rob/%s.jpg" % filename)
    Image.open("/home/rob/%sLOWRES.png" % filename).convert('L').save("/home/rob/%sLOWRES.jpg" % filename)

file_process("costovertime")

show()

