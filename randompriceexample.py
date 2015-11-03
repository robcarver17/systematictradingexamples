from commonrandom import arbitrary_timeseries, generate_trendy_price
from matplotlib.pyplot import show


ans=arbitrary_timeseries(generate_trendy_price(Nlength=180, Tlength=30, Xamplitude=10.0, Volscale=0.0))
print ans
print len(ans)
ans.plot()
show()
