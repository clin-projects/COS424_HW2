__author__ = 'liangshengzhang'

import process as pr
import numpy as np
import time

start_time = time.time()

chr1 = pr.Data(1)

chr1.read()

read_time = time.time() - start_time

hour, minute, second = pr.time_process(read_time)

print '\n'
print 'Loading time: ' + str(hour) + "h " + str(minute) + "m " + str(second) + "s "

start_time = time.time()
chr1.data_extract(strand_binary=True, pos_normalize=True)

from sklearn import preprocessing
imputer = preprocessing.Imputer(copy=False)
imputer.fit_transform(chr1.train_beta)

process_time = time.time() - start_time
hour, minute, second = pr.time_process(process_time)
print '\n'
print 'Processing time: ' + str(hour) + "h " + str(minute) + "m " + str(second) + "s "

start_time = time.time()
from sklearn import linear_model
clf = linear_model.LinearRegression()

train_X = np.transpose(chr1.train_beta[chr1.sample_not_nan,:])

print '\n'
clf.fit(train_X, np.transpose(chr1.train_beta[chr1.sample_nan,:]))

sample_X = np.transpose(chr1.sample_beta[chr1.sample_not_nan,:])

predict = clf.predict(sample_X)

print np.shape(predict)

"""
# Normalized square error for prediction
err = 0
test_not_nan = []
for n in range(len(predict)):
    if not np.isnan(chr1.test_beta[chr1.sample_nan[n]]):
        err += (predict[n] - chr1.test_beta[chr1.sample_nan[n]])**2
        test_not_nan.append(chr1.sample_nan[n])
err = err / len(test_not_nan)

# Varaince of the test data used for comparison
var = np.var(chr1.test_beta[np.array(test_not_nan)])

print '\n'
print "Prediction Error Square:", err
print "Error percentage:", err/var
"""
