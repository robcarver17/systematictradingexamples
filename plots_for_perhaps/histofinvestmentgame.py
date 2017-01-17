import random
from matplotlib.pyplot import plot, show, xticks, xlabel, ylabel, legend, yscale, title, savefig, rcParams, figure, hist, text, bar, subplots
import matplotlib.pyplot as plt
import Image
import numpy as np

from itertools import cycle

lines = ["-","--","-."]
linecycler = cycle(lines)    
colorcycler=cycle(["red", "blue", "green"])

def linehist(x, color="blue", linestyle="-", bins=10, linewidth=1):
    y,binEdges =np.histogram(x, bins=bins)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    plot(bincenters,y,'-', color=color, linestyle=linestyle, linewidth=linewidth) 

data=[-.6, -.5,
       -.3, -.3, -.3,
      -.2, -.2, 
      -.1,-.1,-.1,-.1,
      0.0, 0.0,0.0, 0.0,0.0,0.0,
      .1, .1,.1,.1,.1,
      .2,.2,.2,
      .3,.3,.3,.3,
      .4,.4,.4,.7, .8, .8]

linehist(data, linewidth=2)
frame=plt.gca()

frame.get_yaxis().set_visible(False)
frame.set_xlim([-0.8, 0.8])
frame.set_xticks([-.8, -.4, 0.0,.4, 0.8])

    


rcParams.update({'font.size': 18})


def file_process(filename):
    fig = plt.gcf()
    fig.set_size_inches(18.5,10.5)
    fig.savefig("/home/rob/%s.png" % filename,dpi=300)
    fig.savefig("/home/rob/%sLOWRES.png" % filename,dpi=50)
    
    Image.open("/home/rob/%s.png" % filename).convert('L').save("/home/rob/%s.jpg" % filename)
    Image.open("/home/rob/%sLOWRES.png" % filename).convert('L').save("/home/rob/%sLOWRES.jpg" % filename)

file_process("investmentgame")
show()

mu=np.mean(data)
sigma=np.std(data)
x = np.linspace(-2, 2, 100)
import matplotlib.mlab as mlab
y=mlab.normpdf(x, mu, sigma)
y=[yvalue * 5.0 for yvalue in y]

linehist(data, linewidth=2)
plt.plot(x,y, linewidth=3, linestyle="dashed", color="gray")

plt.legend(['Actual','Gaussian distribution'])

frame=plt.gca()

frame.get_yaxis().set_visible(False)
frame.set_xlim([-1.2, 1.2])
frame.set_xticks([-1.0, -.5, 0.0,.5, 1.0])


file_process("investmentgame_gauss")
show()


monte_carlo=100000
monte_length=35
avgs=[]
for unused_index in range(monte_carlo):
    returns=[random.gauss(np.mean(data),np.std(data)) for not_used in range(monte_length)]
    avgs.append(np.mean(returns))

avgs.sort()
morethan=[x for x in avgs if x<0.0]
ratio=float(len(morethan))/float(monte_carlo)
print("Proportion %f" % ratio)
print("Std %f" % np.std(avgs))

linehist(avgs, bins=100, linewidth=2)
frame=plt.gca()

frame.get_yaxis().set_visible(False)
frame.set_xlim([-0.1, 0.3])
frame.set_xticks([-.1, 0.0, 0.1,.2, 0.3])

#frame.annotate("Actual return 0.086", xy=(0.086, 12000),xytext=(0.086, 25000), arrowprops=dict(facecolor='black', shrink=0.05))
    

rcParams.update({'font.size': 18})

file_process("ismeandifferent")

show()