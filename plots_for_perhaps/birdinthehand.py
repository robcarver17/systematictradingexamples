import Image
from matplotlib.pyplot import plot, show, xticks, xlabel, ylabel, legend, yscale, title, savefig, rcParams, figure, hist, scatter

import matplotlib.pyplot as plt
import numpy as np

cost=0.012 - 0.005
stdev=0.302 - 0.289

## generate guassian points from benefit
volstructure=[ 1.645*stdev, 1.28*stdev, 0.6666*stdev] 
midbenefit=0.302 - 0.25

x=[0.0, 1.0]

ones = np.ones(2)

fig, ax = plt.subplots()

for i, val in enumerate(volstructure):
    alpha = 0.5*(i+1)/len(volstructure) # Modify the alpha value for each iteration.
    ax.fill_between(x, [-cost, midbenefit-cost+val], [-cost, midbenefit-cost-val], color='red', alpha=alpha)

ax.plot(x, [-cost, midbenefit-cost], color='black', linewidth=3) # Plot the original signal

ax.set_ylim([-cost-np.max(volstructure), midbenefit-cost+np.max(volstructure)+stdev/2.0])
ax.set_yticks([ -cost, 0.0, -cost+midbenefit])
ax.get_xaxis().set_visible(False)
plt.axhline(0.0, linestyle="--")

def file_process(filename):
    fig = plt.gcf()
    fig.set_size_inches(18.5,10.5)
    fig.savefig("/home/rob/%s.png" % filename,dpi=300)
    fig.savefig("/home/rob/%sLOWRES.png" % filename,dpi=50)
    
    Image.open("/home/rob/%s.png" % filename).convert('L').save("/home/rob/%s.jpg" % filename)
    Image.open("/home/rob/%sLOWRES.png" % filename).convert('L').save("/home/rob/%sLOWRES.jpg" % filename)

rcParams.update({'font.size': 18})

file_process("birdinthehanddivbenefit")

plt.show()

