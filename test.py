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

print chr1.data_map["train"][0]
