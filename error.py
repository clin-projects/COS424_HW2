__author__ = 'liangshengzhang'

import numpy as np
import csv
import pylab
import scipy.stats as stats

filename = "regression"
f = open(filename+".txt")
reader = csv.reader(f, delimiter='\t')
data_frame = list(reader)
name = {}
name[(data_frame[0][0]).split(' ')[0]] = 1
name[(data_frame[0][0]).split(' ')[4]] = 5
name[(data_frame[0][0]).split(' ')[7]] = 8

class Data(object):
    def __init__(self):
        self.prediction = []
        self.true_value = []
        self.score = []

data = Data()

for n in range(1,len(data_frame)):
    s = data_frame[n][0].split(' ')
    try:
        if s[0] == '[':
            data.prediction.append(float(s[1][:-1]))
            data.true_value.append(float(s[5]))
            data.score.append(float(s[8]))
        else:
            data.prediction.append(float(s[0][1:-1]))
            data.true_value.append(float(s[4]))
            data.score.append(float(s[7]))
    except ValueError:
        print n, s

data.prediction = np.array(data.prediction)
data.true_value = np.array(data.true_value)
data.score = np.array(data.score)

data.error = data.prediction - data.true_value

font = {'size'   : 18}

pylab.rc('font', **font)

pylab.figure(1)
pylab.hist(data.error, bins = 1000, normed=True)

pylab.savefig(filename+"_prediction_error_hist.pdf",box_inches='tight')

var = np.var(data.error)
print "Var: ", var
data.error = [n / (var**(0.5)) for n in data.error]

pylab.figure(2)
stats.probplot(data.error, dist="norm", plot=pylab)

pylab.savefig(filename+"_prediction_error_standardized_qq_plot_normal.pdf",box_inches='tight')

from scipy import stats
skew = stats.skew(data.error)
kurtosis = 3 + stats.kurtosis(data.error)

# Calculate the Jarque-Bera test for normality
jb = (len(data.error) / 6.) * (skew**2 + (1 / 4.) * (kurtosis - 3)**2)
jb_pv = stats.chi2.sf(jb, 2)

print jb, jb_pv, skew, kurtosis

pylab.show()

