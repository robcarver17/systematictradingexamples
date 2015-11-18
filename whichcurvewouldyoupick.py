"""

Generate two account curves, one better than the other over the long run

When we see one we like for demonstration purposes we 'snapshot' it

We then re-generate the plot year by year


"""

from common import ROOT_DAYS_IN_YEAR, DAYS_IN_YEAR, arbitrary_timeseries, slices_for_ts
from commonrandom import autocorr_skewed_returns
from matplotlib.pyplot import show
import pandas as pd

## set the problem up
rho=0.0
want_skew=0.0
want_Sharpes=[0.5, 1.0]
length_period_years=20

# calculations
length_period=length_period_years*DAYS_IN_YEAR
want_ann_stdev=0.2
want_ann_mean=[ann_SR*want_ann_stdev for ann_SR in want_Sharpes]
want_mean_list=[this_ann_mean/DAYS_IN_YEAR for this_ann_mean in want_ann_mean]
want_stdev=want_ann_stdev/ROOT_DAYS_IN_YEAR

happywithcurve=False

while not happywithcurve:
    ## generate a new Curve
    equitycurves=[arbitrary_timeseries(autocorr_skewed_returns(rho, want_mean,  want_stdev, 
                  want_skew, size=length_period), index_start=pd.datetime(1995,1,1))
                  for want_mean in want_mean_list]    
    equitycurves_pd=pd.concat(equitycurves, axis=1)
    equitycurves_pd.columns=["A", "B"]
    equitycurves_pd.cumsum().plot()
    show()
    ans=raw_input("Happy with this? Y / N? ")
    if ans=="Y":
        happywithcurve=True

sliced_data=slices_for_ts(equitycurves[0])
start_point=sliced_data[0]

for idx in range(len(sliced_data))[1:]:
    sliced_curves=[eq_curve[start_point:sliced_data[idx]] for eq_curve in equitycurves]
    equitycurves_pd=pd.concat(sliced_curves, axis=1)
    equitycurves_pd.columns=["A", "B"]
    equitycurves_pd.cumsum().plot()
    show()

