'''
使用蒙克卡罗法，利用pyspark来估算π的大小
概率预估
函数式变成，简化代码
''' 

import numpy as np
import operator
from pyspark import SparkConf
from pyspark import SparkContext
import pyspark

import os
os.environ["PYSPARK_PYTHON"]="/Library/Frameworks/Python.framework/Versions/3.7/bin/python3.7"
# ================================================================
total = int(100000)
# parallelize a data set into the cluster
# version 1
# rdd = SparkContext().parallelize(range(1, total))\
#                     .setName("parallelized_data").cache()\
#                     .map(lambda x: (np.random.random(), np.random.random())) \
#                     .map(lambda x: 1 if (x[0]**2 + x[1]**2) < 1 else 0)\
#                     .reduce(lambda x, y: x + y) / float(total) * 4

# version 2
rdd = SparkContext().parallelize(range(1,total))\
                    .map(lambda x: 1 if sum(np.random.random(2) ** 2) < 1 else 0)\
                    .reduce(lambda x, y: x + y)\
                    / float(total) * 4


print("iterate {} time".format(total))
print("estimated pi : {}".format(rdd))
