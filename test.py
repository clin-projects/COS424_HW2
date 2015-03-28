__author__ = 'liangshengzhang'

import process as pr
import numpy as np
import time
from math import log, exp

start_time = time.time()

chr1 = pr.Data(1)

chr1.read(detail=True)

read_time = time.time() - start_time

hour, minute, second = pr.time_process(read_time)

print '\n'
print 'Loading time: ' + str(hour) + "h " + str(minute) + "m " + str(second) + "s "

start_time = time.time()
chr1.data_extract(strand_binary=True, pos_normalize=True)

start_time = time.time()
from sklearn import preprocessing
imputer = preprocessing.Imputer(copy=False)
imputer.fit_transform(chr1.train_beta)

process_time = time.time() - start_time
hour, minute, second = pr.time_process(process_time)
print '\n'
print 'Processing time: ' + str(hour) + "h " + str(minute) + "m " + str(second) + "s "

from sklearn import mixture
train_data = np.mean(chr1.train_beta[chr1.sample_not_nan,:],axis=1)
sample_data = chr1.sample_beta[chr1.sample_not_nan]

cluster_num = 8
gmm = mixture.GMM(n_components = cluster_num)
gmm.fit(train_data)
print gmm.bic(train_data)

train_proba = gmm.predict_proba(train_data)

train_X = np.zeros((len(chr1.train_beta[0]),cluster_num))

cluster_prob = np.zeros((len(chr1.train_beta[0]),cluster_num))

for n in range(len(chr1.sample_not_nan)):
    pos = chr1.sample_not_nan[n]
    for x in range(len(chr1.train_beta[0])):
        beta = chr1.train_beta[pos][x]
        for i in range(cluster_num):
            train_X[x][i] += train_proba[n][i] * beta
            cluster_prob[x][i] += train_proba[n][i]

for x in range(len(chr1.train_beta[0])):
        for i in range(cluster_num):
            train_X[x][i] /= cluster_prob[x][i]

print cluster_prob[0]
print train_X[0]

sample_proba = gmm.predict_proba(sample_data)

sample_X = np.zeros(cluster_num)
cluster_prob = np.zeros(cluster_num)
for n in range(len(chr1.sample_not_nan)):
    pos = chr1.sample_not_nan[n]
    beta = chr1.sample_beta[pos]
    for i in range(cluster_num):
        sample_X[i] += beta * sample_proba[n][i]
        cluster_prob[i] += sample_proba[n][i]

sample_X = [sample_X[n]/cluster_prob[n] for n in range(cluster_num)]
print sample_X

cluster_time = time.time() - start_time
hour, minute, second = pr.time_process(cluster_time)
print '\n'
print 'Clustering time: ' + str(hour) + "h " + str(minute) + "m " + str(second) + "s "

start_time = time.time()
from sklearn import linear_model

predict = []
clf = linear_model.LinearRegression()
clf.fit(train_X, chr1.train_beta[0,:])
predict.append(clf.predict(sample_X))

print "Prediction:", predict[0]
print "True:", chr1.test_beta[chr1.sample_nan[0]]

predict_time = time.time() - start_time
hour, minute, second = pr.time_process(predict_time)
print '\n'
print 'Fitting and Predicting time: ' + str(hour) + "h " + str(minute) + "m " + str(second) + "s "

"""
import pylab
pylab.plot(pca.explained_variance_ratio_)
pylab.show()
"""