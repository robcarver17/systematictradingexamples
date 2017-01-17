from optimisation import *
from matplotlib.pyplot import plot, show, xticks, xlabel, ylabel, legend, yscale, title, savefig, rcParams, figure, hist, text, bar, subplots
import matplotlib.pyplot as plt
import Image
import pandas as pd

from itertools import cycle, islice

lines = ["-","--","-."]

filename="/home/rob/stuff/Writing/NonFiction/Quant/perhaps/pictures/tradingCostsDirectIndirect.csv"

results=pd.read_csv(filename)
results.columns=[""]+list(results.columns[1:])
results.index=results.iloc[:,0]

linecycler = cycle(lines)    
colorcycler=cycle(["red", "blue", "green"])


plt.subplot(2,1,1)
toplot=results["FIXED"]
toplot.plot(kind="bar")
ylabel("Fixed cost % per year")

plt.grid(False)
frame=plt.gca()
frame.set_ylim([-0.1, 1.5])
frame.set_yticks([ 0.0, 0.5, 1.0])
frame.get_xaxis().set_ticks_position('none')
#for tick in frame.get_xaxis().get_majorticklabels():
#    tick.set_horizontalalignment("center")
frame.set_xticklabels([])


linecycler = cycle(lines)    
colorcycler=cycle(["red", "blue", "green"])

plt.subplot(2,1,2)
toplot=results.CPT
ans=toplot.plot(kind="bar")
ylabel("Cost per turnover %")
plt.grid(False)
frame=plt.gca()

frame.set_ylim([-0.1, 1.5])
frame.set_yticks([ 0.0, 0.5, 1.0])
frame.get_xaxis().set_ticks_position('none')
frame.set_xticklabels([])
plt.text(0.4, -0.4, results.index[0])
plt.text(1.4, -0.4, results.index[1])
plt.text(2.5, -0.4, results.index[2])

#for tick in frame.get_xaxis().get_majorticklabels():
#    tick.set_horizontalalignment("center")


#frame.get_xaxis().set_visible(False)
#frame.get_yaxis().set_visible(False)

#frame.set_xticks([-0.9, 0.0, 0.9])

rcParams.update({'font.size': 18})



def file_process(filename):
    fig = plt.gcf()
    fig.set_size_inches(18.5,10.5)
    fig.savefig("/home/rob/%s.png" % filename,dpi=300)
    fig.savefig("/home/rob/%sLOWRES.png" % filename,dpi=50)
    
    Image.open("/home/rob/%s.png" % filename).convert('L').save("/home/rob/%s.jpg" % filename)
    Image.open("/home/rob/%sLOWRES.png" % filename).convert('L').save("/home/rob/%sLOWRES.jpg" % filename)

file_process("costsdirectindirect")


show()

filename="/home/rob/stuff/Writing/NonFiction/Quant/perhaps/pictures/tradingCostsSize.csv"

results=pd.read_csv(filename)
results.columns=[""]+list(results.columns[1:])
results.index=results.iloc[:,0]

linecycler = cycle(lines)    
colorcycler=cycle(["red", "blue", "green"])


plt.subplot(2,1,1)
toplot=results[results.Country=="US"].FIXED
toplot.plot(linestyle=next(linecycler), color=next(colorcycler), linewidth=3)
toplot=results[results.Country=="UK"].FIXED
toplot.plot(linestyle=next(linecycler), color=next(colorcycler), linewidth=3)

ylabel("Fixed cost % per year")

plt.legend(["US", "UK"],loc="upper right")

plt.grid(False)
frame=plt.gca()
frame.set_ylim([-0.01, 0.075])
frame.set_yticks([ 0.0, 0.03, 0.06])
frame.get_xaxis().set_ticks_position('none')
current_xlim=frame.get_xlim()
frame.set_xlim([current_xlim[0]-0.1, current_xlim[1]+.1])
#for tick in frame.get_xaxis().get_majorticklabels():
#    tick.set_horizontalalignment("center")
frame.set_xticklabels([])


linecycler = cycle(lines)    
colorcycler=cycle(["red", "blue", "green"])

plt.subplot(2,1,2)
toplot=results[results.Country=="US"].CPT
toplot.plot(linestyle=next(linecycler), color=next(colorcycler), linewidth=3)
toplot=results[results.Country=="UK"].CPT
toplot.plot(linestyle=next(linecycler), color=next(colorcycler), linewidth=3)
ylabel("Cost per turnover %")
plt.grid(False)
frame=plt.gca()

frame.set_ylim([-0.1, 1.5])
frame.set_yticks([ 0.0, 0.5, 1.0])
frame.get_xaxis().set_ticks_position('none')
current_xlim=frame.get_xlim()
frame.set_xlim([current_xlim[0]-0.1, current_xlim[1]+0.1])
frame.set_xticklabels([])
plt.text(-0.1, -0.4, results.index[0])
plt.text(0.9, -0.4, results.index[1])
plt.text(1.9, -0.4, results.index[2])
plt.text(2.9, -0.4, results.index[3])
plt.text(3.9, -0.4, results.index[4])


#for tick in frame.get_xaxis().get_majorticklabels():
#    tick.set_horizontalalignment("center")


#frame.get_xaxis().set_visible(False)
#frame.get_yaxis().set_visible(False)

#frame.set_xticks([-0.9, 0.0, 0.9])

rcParams.update({'font.size': 18})

file_process("costsbysize")

show()


filename="/home/rob/stuff/Writing/NonFiction/Quant/perhaps/pictures/tradingCostsByCountry_FIXED.csv"

results1=pd.read_csv(filename)
results1.columns=[""]+list(results1.columns[1:])
results1.index=results1.iloc[:,0]

filename="/home/rob/stuff/Writing/NonFiction/Quant/perhaps/pictures/tradingCostsByCountry_CPT.csv"

results2=pd.read_csv(filename)
results2.columns=[""]+list(results2.columns[1:])
results2.index=results2.iloc[:,0]


my_colors = list(islice(cycle(['b', 'r', 'k']), None, len(results2)))
ax=results1.plot(kind="bar",  color=my_colors)
ax.set_xticklabels(["UK", "German", "US"], rotation=0)

ylabel("Fixed cost % per year")


plt.grid(False)

rcParams.update({'font.size': 18})

file_process("costsbycountryFIXED")

show()


ax=results2.plot(kind="bar", color=my_colors)


ylabel("Cost per turnover %")
ax.set_xticklabels(["UK", "German", "US"], rotation=0)
plt.grid(False)




rcParams.update({'font.size': 18})

file_process("costsbycountryCPT")

show()


filename="/home/rob/stuff/Writing/NonFiction/Quant/perhaps/pictures/tradingCostsMarketCapFIXED.csv"

results1=pd.read_csv(filename)
results1.columns=[""]+list(results1.columns[1:])
results1.index=results1.iloc[:,0]

filename="/home/rob/stuff/Writing/NonFiction/Quant/perhaps/pictures/tradingCostsMarketCapCPT.csv"

results2=pd.read_csv(filename)
results2.columns=[""]+list(results2.columns[1:])
results2.index=results2.iloc[:,0]


my_colors = list(islice(cycle(['b', 'r', 'k']), None, len(results2)))
ax=results1.plot(kind="bar",  color=my_colors)
ax.set_xticklabels(["Large cap", "Mid cap", "Small cap"], rotation=0)

ylabel("Fixed cost % per year")


plt.grid(False)

rcParams.update({'font.size': 18})

file_process("costsbycapFIXED")

show()


ax=results2.plot(kind="bar", color=my_colors)


ylabel("Cost per turnover %")
ax.set_xticklabels(["Large cap", "Mid cap", "Small cap"], rotation=0)
plt.grid(False)




rcParams.update({'font.size': 18})

file_process("costsbycapCPT")

show()

