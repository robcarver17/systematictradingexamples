import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, show, xticks, xlabel, ylabel, legend, yscale, title, savefig, rcParams, figure, hist, text, bar, subplots
import Image
import pandas as pd

def file_process(filename):
    fig = plt.gcf()
    fig.set_size_inches(18.5,10.5)
    fig.savefig("/home/rob/%s.png" % filename,dpi=300)
    fig.savefig("/home/rob/%sLOWRES.png" % filename,dpi=50)
    
    Image.open("/home/rob/%s.png" % filename).convert('L').save("/home/rob/%s.jpg" % filename)
    Image.open("/home/rob/%sLOWRES.png" % filename).convert('L').save("/home/rob/%sLOWRES.jpg" % filename)

costsnotrading=pd.read_csv("/home/rob/workspace/systematictradingexamples/plots_for_perhaps/etfvssharesbreakeven.csv")

costsnotrading.index=costsnotrading.Value
costsnotrading=costsnotrading.drop("Value", 1)

costsnotrading.plot(style=['b-','g--.'])

frame=plt.gca()
frame.set_ylim([0.0, 0.3])
frame.set_yticks([ 0.1, 0.2, 0.3])

frame.set_xticks([10000, 15000, 20000])



rcParams.update({'font.size': 18})
file_process("etfsharesbreakeven")

show()


costsnotrading=pd.read_csv("/home/rob/workspace/systematictradingexamples/plots_for_perhaps/etfvssharesbreakevenwithtrading.csv")

costsnotrading.index=costsnotrading.Value
costsnotrading=costsnotrading.drop("Value", 1)

costsnotrading.plot(style=['b-','g--.'])

frame=plt.gca()
frame.set_ylim([0.0, 0.6])
frame.set_yticks([ 0.2, 0.4, 0.6])

frame.set_xticks([100000, 200000, 300000, 400000, 500000])



rcParams.update({'font.size': 18})
file_process("etfsharesbreakevenwithtrading")

show()

