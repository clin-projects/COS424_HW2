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

train_beta_mean = np.mean(chr1.train_beta,axis=1)

# Normalized square error for prediction
err = 0
test_not_nan = []
for n in range(len(chr1.sample_nan)):
    if not np.isnan(chr1.test_beta[chr1.sample_nan[n]]):
        err += (train_beta_mean[chr1.sample_nan[n]] - chr1.test_beta[chr1.sample_nan[n]])**2
        test_not_nan.append(chr1.sample_nan[n])
err = err / len(test_not_nan)

# Varaince of the test data used for comparison
var = np.var(chr1.test_beta[np.array(test_not_nan)])

print '\n'
print "Prediction Error Square:", err
print "Error percentage:", err/var

