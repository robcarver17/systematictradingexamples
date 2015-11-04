"""
See http://qoppac.blogspot.co.uk/2015/11/using-random-data.html for more examples
"""

from commonrandom import arbitrary_timeindex, skew_returns_annualised
from matplotlib.pyplot import show, hist
from common import DAYS_IN_YEAR, ROOT_DAYS_IN_YEAR, account_curve
import pandas as pd
import numpy as np
import scipy.stats as st

def plot_random_curves(pddf_rand_data):

    pddf_rand_data.cumsum().plot(colormap="Pastel2", legend=False)
    
    avg_random=pddf_rand_data.cumsum().mean(axis=1)
    avg_random.plot(color="k")




"""
Set up the parameters
"""

length_backtest_years=10
number_of_random_curves=1000
annualSR=0.5
want_skew=1.0


"""
Do the bootstrap of many random curves
"""

length_bdays=length_backtest_years*DAYS_IN_YEAR

random_curves=[skew_returns_annualised(annualSR=annualSR, want_skew=want_skew, size=length_bdays) 
               for NotUsed in range(number_of_random_curves)]

## Turn into a dataframe

random_curves_npa=np.array(random_curves).transpose()
pddf_rand_data=pd.DataFrame(random_curves_npa, index=arbitrary_timeindex(length_bdays), columns=[str(i) for i in range(number_of_random_curves)])

## This is a nice representation as well
acccurves_rand_data=[account_curve(pddf_rand_data[x]) for x in pddf_rand_data]

"""
Plot the curves, just for interest
"""

plot_random_curves(pddf_rand_data)
show()

"""
Get the statistic we are interested in

Change 'function_to_apply' to get what you want

"""

function_to_apply=np.percentile
function_args=(100.0/DAYS_IN_YEAR,)
results=pddf_rand_data.apply(function_to_apply, axis=0, args=function_args)

## Annualise (not always needed, depends on the statistic)
#results=[x*ROOT_DAYS_IN_YEAR for x in results]


hist(results, 100)
show()


"""
Or we can do this kind of thing
"""

results=[x.worst_drawdown() for x in acccurves_rand_data]
hist(results, 100)
show()
