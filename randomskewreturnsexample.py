from commonrandom import arbitrary_timeseries, skew_returns_annualised
from matplotlib.pyplot import show

ans=arbitrary_timeseries(skew_returns_annualised(annualSR=1.0, want_skew=0.0, voltarget=0.20, size=10000))
ans.plot()
show()
