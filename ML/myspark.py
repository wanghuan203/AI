'''
使用蒙克卡罗法，利用pyspark来估算π的大小
概率预估
'''

import numpy as np
import operator
from pyspark import SparkContext

import os
os.environ["PYSPARK_PYTHON"]="/Library/Frameworks/Python.framework/Versions/3.7/bin/python3.7"
# ================================================================
# 加载数据集
total = int(100000)
local_collection = range(1,total)

sc = SparkContext()

# parallelize a data set into the cluster
rdd = sc.parallelize(local_collection).setName("parallelized_data").cache()
print(rdd)

# =================================================================
# 处理数据集
# random point 生成随机的点
def map_func(element):
    x = np.random.random()
    y = np.random.random()   ##[0,1)
    return(x, y)


def map_fun_2(element):
    x, y = element
    return 1 if x**2 + y**2 < 1 else 0


rdd2 = rdd.map(map_func).setName("random_point").cache()
# 计算点的数据是否在圆里
rdd3 = rdd2.map(map_fun_2).setName("points_in_out_circle").cache()

# ==================================================================
# 结果展示
# how many point are in the circle
in_circle = rdd3.reduce(operator.add)
pi = 4. * in_circle / total
print("iterate {} time".format(total))
print("estimated pi : {}".format(pi))

