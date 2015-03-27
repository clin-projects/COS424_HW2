__author__ = 'liangshengzhang'

import process as pr
import numpy as np
import time
from math import log,exp

start_time = time.time()

chr1 = pr.Data(1)

chr1.read(detail=True)

read_time = time.time() - start_time

hour, minute, second = pr.time_process(read_time)

print '\n'
print 'Loading time: ' + str(hour) + "h " + str(minute) + "m " + str(second) + "s "

start_time = time.time()
chr1.data_extract(strand_binary=True, pos_normalize=True)
read_time = time.time() - start_time
hour, minute, second = pr.time_process(read_time)
print '\n'
print 'Processing time: ' + str(hour) + "h " + str(minute) + "m " + str(second) + "s "

from sklearn import preprocessing
imputer = preprocessing.Imputer(copy=False)
imputer.fit_transform(chr1.train_beta)

from sklearn import linear_model
clf = linear_model.LinearRegression()

train_X = np.empty([len(chr1.train_start),3])
train_X[:,0] = chr1.train_start
train_X[:,1] = chr1.train_strand
train_X[:,2] = chr1.train_chip

train_beta_fit = np.mean(chr1.train_beta, axis = 1)
train_beta_fit = [log(1.0/n - 1) for n in train_beta_fit]

print '\n'
clf.fit(train_X, train_beta_fit)
print "slope:", clf.intercept_
print "coefficient:", clf.coef_
print "R^2:", clf.score(train_X, train_beta_fit)

# There may be 2 clusters for starting positions

sample_X = np.empty([len(chr1.sample_nan),3]) # end - start is the same
sample_X[:,0] = chr1.sample_start[chr1.sample_nan]
sample_X[:,1] = chr1.sample_strand[chr1.sample_nan]
sample_X[:,2] = chr1.sample_chip[chr1.sample_nan]

predict = clf.predict(sample_X)

# Normalized square error for prediction
err = 0
test_not_nan = []
for n in range(len(predict)):
    if not np.isnan(chr1.test_beta[chr1.sample_nan[n]]):
        err += ((1.0/(1.0+exp(predict[n]))) - chr1.test_beta[chr1.sample_nan[n]])**2
        test_not_nan.append(chr1.sample_nan[n])
err = err / len(test_not_nan)

# Varaince of the test data used for comparison
var = np.var(chr1.test_beta[np.array(test_not_nan)])

print '\n'
print "Prediction Error Square:", err
print "Error percentage:", err/var





