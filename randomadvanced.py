from commonrandom import autocorr_skewed_returns, adj_moments_for_rho
from common import arbitrary_timeseries, autocorr, ROOT_DAYS_IN_YEAR
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

"""
This code verifies that the single frequency method for generating returns with a given distribution 
skew, autocorrelation; all works...

"""

monte_runs=1000

for want_rho in np.arange(-0.8, 0.8, 0.1):
    for want_sharpe in [0.0, 0.5, 1.0, 2.0]:
        for want_stdev in [1.0]:
            for want_skew in [-2.0, -1.0, 0.0, 0.5, 1.0]:
                               
                want_mean=want_stdev*want_sharpe/16.0

                ## autocorrelation introduces biases
                
                (adj_want_mean, adj_want_skew, adj_want_stdev)=adj_moments_for_rho(want_rho, want_mean, want_skew, want_stdev)
                
                manyans=[autocorr_skewed_returns(want_rho, adj_want_mean, adj_want_stdev, adj_want_skew) for notused in range(monte_runs)]
                sample_mean=np.mean([np.mean(x) for x in manyans])
                sample_std=np.mean([np.std(x) for x in manyans])
                sample_skew=np.mean([st.skew(x) for x in manyans])
                sample_rho=np.mean([autocorr(x) for x in manyans])
                
                print "****************************"
                print "Mean, wanted %.2f got %.6f" % (want_mean, sample_mean)
                print "Stdev, wanted %.2f got %.4f" % (want_stdev, sample_std)
                print "Skew, wanted %.2f got %.4f" % (want_skew, sample_skew)
                print "Rho, wanted %.2f got %.4f" % (want_rho, sample_rho)


