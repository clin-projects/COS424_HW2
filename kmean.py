__author__ = 'liangshengzhang'

import process as pr
import numpy as np
import time
import sys

start_time = time.time()

cluster_num = int(sys.argv[1])
print "Number of cluster:", cluster_num

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

from sklearn import cluster
train_data = chr1.train_beta[chr1.sample_not_nan,:]
sample_data = chr1.sample_beta[chr1.sample_not_nan]

kmean = cluster.KMeans(n_clusters = cluster_num)
labels = kmean.fit_predict(train_data)

train_X = np.transpose(kmean.cluster_centers_)

sample_center = [[] for x in range(cluster_num)]

for n in range(len(labels)):
    sample_center[labels[n]].append(chr1.sample_not_nan[n])

sample_X = []
for n in sample_center:
    sample_X.append(np.mean(chr1.sample_beta[np.array(n)]))

cluster_time = time.time() - start_time
hour, minute, second = pr.time_process(cluster_time)
print '\n'
print 'Clustering time: ' + str(hour) + "h " + str(minute) + "m " + str(second) + "s "

start_time = time.time()
from sklearn import linear_model

predict = []
clf = linear_model.LinearRegression()
score = []
if cluster_num < 20:
    coef = []
    intercept = []
else:
    coef = None
    intercept = None
chr1.regression(clf, train_X, sample_X, predict, score, intercept=intercept, coef=coef)

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
filename = "kmean_" + str(cluster_num) + ".txt"

chr1.output(filename, predict_not_nan, predict, true_val, score = score, intercept=intercept, coef=coef)

output_time = time.time() - start_time
hour, minute, second = pr.time_process(output_time)
print '\n'
print 'Output time: ' + str(hour) + "h " + str(minute) + "m " + str(second) + "s "

