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

predict = []
score = []
train_X = np.transpose(chr1.train_beta[chr1.sample_not_nan,:])
sample_X = chr1.sample_beta[chr1.sample_not_nan]
clf = linear_model.LinearRegression()

chr1.regression(clf, train_X, sample_X, predict, score)

predict_time = time.time() - start_time
hour, minute, second = pr.time_process(predict_time)
print '\n'
print 'Fitting and Predicting time: ' + str(hour) + "h " + str(minute) + "m " + str(second) + "s "


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
filename = "regression.txt"

chr1.output(filename, predict_not_nan, predict, true_val, score = score)

output_time = time.time() - start_time
hour, minute, second = pr.time_process(output_time)
print '\n'
print 'Output time: ' + str(hour) + "h " + str(minute) + "m " + str(second) + "s "

