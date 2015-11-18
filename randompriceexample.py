"""
See http://qoppac.blogspot.co.uk/2015/11/using-random-data.html for more examples
"""


from common import arbitrary_timeseries
from commonrandom import  generate_trendy_price
from matplotlib.pyplot import show


ans=arbitrary_timeseries(generate_trendy_price(Nlength=180, Tlength=30, Xamplitude=10.0, Volscale=0.0))
print ans
print len(ans)
ans.plot()
show()

ans=arbitrary_timeseries(generate_trendy_price(Nlength=180, Tlength=30, Xamplitude=10.0, Volscale=0.1))
ans.plot()
show()

ans=arbitrary_timeseries(generate_trendy_price(Nlength=180, Tlength=30, Xamplitude=10.0, Volscale=0.5))
ans.plot()
show()
