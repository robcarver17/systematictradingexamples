"""
See http://qoppac.blogspot.co.uk/2015/11/using-random-data.html for more examples
"""


from commonrandom import arbitrary_timeseries, skew_returns_annualised
from common import cum_perc
from matplotlib.pyplot import show

ans=arbitrary_timeseries(skew_returns_annualised(annualSR=0.5, want_skew=0.0, size=2500))
cum_perc(ans).plot()
show()

ans=arbitrary_timeseries(skew_returns_annualised(annualSR=0.5, want_skew=1.0, size=2500))
cum_perc(ans).plot()
show()

ans=arbitrary_timeseries(skew_returns_annualised(annualSR=1.0, want_skew=-3.0, size=2500))
cum_perc(ans).plot()
show()

