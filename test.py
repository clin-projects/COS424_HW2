__author__ = 'liangshengzhang'

import process as pr
import numpy as np
import time

start_time = time.time()

chr1 = pr.Data(1)

chr1.read(detail=True)

read_time = time.time() - start_time

hour, minute, second = pr.time_process(read_time)

print '\n'
print 'Loading time: ' + str(hour) + "h " + str(minute) + "m " + str(second) + "s "

start_time = time.time()
chr1.data_extract(True)
read_time = time.time() - start_time
hour, minute, second = pr.time_process(read_time)
print '\n'
print 'Processing time: ' + str(hour) + "h " + str(minute) + "m " + str(second) + "s "

from sklearn import linear_model
clf = linear_model.LinearRegression()

from sklearn import preprocessing
imputer = preprocessing.Imputer(copy=False)
imputer.fit_transform(chr1.train_beta)

print np.mean(chr1.train_beta)

#beta_mean = np.mean(chr1.train[4:36])
