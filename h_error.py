__author__ = 'liangshengzhang'

import numpy as np
import csv
import pylab
import scipy.stats as stats

filename = "h_regression"
data_frame = np.loadtxt(filename+".txt", skiprows=1)

class Data(object):
    def __init__(self):
        self.prediction = []
        self.true_value = []
        self.score = []

data = Data()

data.prediction = np.array(data_frame[:,0])
data.true_value = np.array(data_frame[:,1])

data.error = data.prediction - data.true_value

font = {'size'   : 18}

pylab.rc('font', **font)

pylab.figure(1)
pylab.hist(data.error, bins = 1000)

var = np.var(data.error)
print "Var: ", var
data.error = [n / (var**(0.5)) for n in data.error]

pylab.savefig(filename+"_prediction_error_hist.pdf",box_inches='tight')

pylab.figure(2)
stats.probplot(data.error, dist="norm", plot=pylab)
pylab.savefig(filename+"_prediction_error_hist.pdf",box_inches='tight')

from scipy import stats
skew = stats.skew(data.error)
kurtosis = 3 + stats.kurtosis(data.error)

# Calculate the Jarque-Bera test for normality
jb = (len(data.error) / 6.) * (skew**2 + (1 / 4.) * (kurtosis - 3)**2)
jb_pv = stats.chi2.sf(jb, 2)

print jb, jb_pv, skew, kurtosis

pylab.show()

