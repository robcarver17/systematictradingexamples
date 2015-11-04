"""
See http://qoppac.blogspot.co.uk/2015/11/using-random-data.html for more examples
"""


from matplotlib.pyplot import show
from commonrandom import threeassetportfolio

SRlist=[.5, 1.0, 0.0]
clist=[.9,.5,-0.5]

ans=threeassetportfolio(plength=5000, SRlist=SRlist, clist=clist)
ans.cumsum().plot()
show()

print "Standard deviation"

print ans.std()

print "Correlation"
print ans.corr()

print "Mean"
print ans.mean()

print "Annualised SR"

print 16*ans.mean()/ans.std()