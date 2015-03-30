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

predict = train_beta_mean[chr1.sample_nan]

start_time = time.time()
# Normalized square error for prediction
test_not_nan = []
predict_not_nan = []
true_val = []
err, var= chr1.error_metric(predict, test_not_nan, predict_not_nan, true_val)

print '\n'
print "Number of points:", len(test_not_nan)
print "Var:", var
print "Prediction Error Square:", err
print "Error percentage:", err/var

# Only print out values which have true values
filename = "mean.txt"

chr1.output(filename, predict_not_nan, predict, true_val)

output_time = time.time() - start_time
hour, minute, second = pr.time_process(output_time)
print '\n'
print 'Output time: ' + str(hour) + "h " + str(minute) + "m " + str(second) + "s "

