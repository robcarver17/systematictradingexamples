import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, show, xticks, xlabel, ylabel, legend, yscale, title, savefig, rcParams, figure, hist, text, bar, subplots
import Image

def variance_f(sigma):
    weights=[1.0/sigma.shape[0]]*sigma.shape[0]
    return 1.0/((np.matrix(weights)*sigma*np.matrix(weights).transpose())[0,0]**.5)

def make_corr(dimension, offdiag=0.0):

    corr=np.array([[offdiag]*dimension]*dimension)
    corr[np.diag_indices(dimension)]=1.0
    return corr


#cfactors=[0.1, 0.5, 0.6, 0.75,  0.8]
#ndimlist=[1, 2, 3,4,5, 11, 15, 20]
#maxdim=[3,      4,   10, 15, 20 ]
cfactors=[.75]
ndimlist=[1,11]
maxdim=[100]

#costs=[0.0005, 0.0033]
costs=[0]*len(ndimlist)

applyzero=[False, False, False, False, False, False, False, False, False]

basestd=0.27
basearithmean=0.05
riskfree=0.000

results_sr=[]
results_gmm=[]
results_std=[]

for (cidx, cfactor) in enumerate(cfactors):
    smallres_sr=[]
    smallres_gmm=[]
    smallres_std=[]

    for nx, ndim in enumerate(ndimlist):
        if ndim<=maxdim[cidx]:
            if applyzero[cidx]:
                gsr=0.0
                gmm=0.0
                new_std=basestd
            else:
                div_factor=variance_f(make_corr(ndim, cfactor))
                new_std=basestd/div_factor
                variance=new_std**2
                gmm=basearithmean- costs[nx] - variance/2.0
                gsr=(gmm - riskfree) / new_std
                
            smallres_sr.append(gsr)
            smallres_gmm.append(gmm*100.0)
            smallres_std.append(new_std)
        
        print("Cfactor %f ndim %d GMM %f STD %f SR %f" % (cfactor, ndim, gmm, new_std, gsr))
          
    results_sr.append(smallres_sr)
    results_gmm.append(smallres_gmm)
    results_std.append(smallres_std)


from itertools import cycle

lines =           ["-",   "--",   "-.",     ":", "-"]
linecycler = cycle(lines)    
colorcycler=cycle(["red", "blue", "green", "red", "blue"])

for r in range(len(results_sr)):
    plot(results_sr[r], color=next(colorcycler), linestyle=next(linecycler), linewidth=3)
    
#xticks(range(len(ndimlist))[0::2], ndimlist[0::2])
xticks(range(len(ndimlist)), ndimlist)
legend(cfactors, loc="top left", title="Correlations", prop={'size': 18})
#title("Diversification for various average correlations")
ylabel("Sharpe ratio")
xlabel("Number of assets")

frame=plt.gca()
frame.set_ylim([0.05, 0.25])
frame.set_yticks([ 0.05, 0.10, 0.15, 0.20, 0.25])


rcParams.update({'font.size': 18})
ax=plt.gca()
ax.get_legend().get_title().set_fontsize('18')

def file_process(filename):
    fig = plt.gcf()
    fig.set_size_inches(18.5,10.5)
    fig.savefig("/home/rob/%s.png" % filename,dpi=300)
    fig.savefig("/home/rob/%sLOWRES.png" % filename,dpi=50)
    
    Image.open("/home/rob/%s.png" % filename).convert('L').save("/home/rob/%s.jpg" % filename)
    Image.open("/home/rob/%sLOWRES.png" % filename).convert('L').save("/home/rob/%sLOWRES.jpg" % filename)

file_process("divbenefit_sr_all_unstacked")

show()


for r in range(len(results_gmm)):
    plot(results_gmm[r], color=next(colorcycler), linestyle=next(linecycler), linewidth=3)
    
#xticks(range(len(ndimlist))[0::2], ndimlist[0::2])
xticks(range(len(ndimlist)), ndimlist)
legend(cfactors, loc="top left", title="Correlations", prop={'size': 18})
#title("Diversification for various average correlations")
ylabel("Geometric mean %")
xlabel("Number of assets")

rcParams.update({'font.size': 18})
ax=plt.gca()
ax.get_legend().get_title().set_fontsize('18')

file_process("divbenefit_gmm_all_unstacked")

show()


for r in range(len(results_std)):
    plot(results_std[r], color=next(colorcycler), linestyle=next(linecycler), linewidth=3)
    
#xticks(range(len(ndimlist))[0::2], ndimlist[0::2])
xticks(range(len(ndimlist)), ndimlist)
legend(cfactors, loc="top left", title="Correlations", prop={'size': 18})
#title("Diversification for various average correlations")
ylabel("Standard deviation")
xlabel("Number of assets")

rcParams.update({'font.size': 18})
ax=plt.gca()
ax.get_legend().get_title().set_fontsize('18')

file_process("divbenefit_std_all_unstacked")

show()
