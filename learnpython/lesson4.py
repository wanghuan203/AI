'''
@author: whuan
@contact:
@file: lesson4.py
@time: 2018/10/21 20:24
@desc:数据分析,通用函数
ufunc是对ndarray中的数据执行元素级运算的函数
'''
import numpy as np
from numpy.random import randn

arr = np.arange(10)
print(np.sqrt(arr))

x = np.random.randn(8)
y = np.random.randn(8)
print(x)
print(y)

# 返回多个数据对应位置的最大值,
print(np.maximum(x, y))

# fen li zheng shu he xiao shu
print(np.modf(x))

points = np.arange(-5, 5, 0.01)
xs, ys = np.meshgrid(points, points)
import matplotlib.pyplot as plt

z = np.sqrt(xs ** 2 + ys ** 2)
print(z)
plt.imshow(z, cmap=plt.cm.gray)
# plt.imshow(z, cmap=plt.cm.spring)

plt.colorbar()
plt.title("ping fang")
# plt.show()

# tiao jian luoji

xarr = np.array([1, 2, 3, 4, 5])
yarr = np.array([2, 3, 4, 5, 6])
cond = np.array([True, False, True, True, False])

result1 = [(x if c else y)
           for x, y, c in zip(xarr, yarr, cond)]

print(result1)

result2 = np.where(cond, xarr, yarr)
print(result2)

arr = np.random.randn(4, 4)
np.where(arr > 0, 2, -2)
print(np.where(arr > 0, 2, arr))

# np.where(cond1 & cond2,0,np.where(cond1,1,np.where(cond2,2,3)))

#数学与统计方法
arr = randn(5, 4)
arr.mean()
np.mean(arr)
arr.sum()
arr.mean(axis=1)

arr=randn(100)
(arr>0).sum(0)

arr= np.array([[0,1,2],[3,4,5],[6,7,8]])
#列相加，行乘法
print(arr.cumsum(0))
print(arr.cumprod(1))

#用于布尔型数组的方法
arr= randn(100)
print((arr>0).sum())

bools = np.array([False,False,True,False])
#任何一个true就是true
print(bools.any())
#全都是true才是true
print(bools.all())

#排序
arr=randn(8)
print(arr)
arr.sort()
print(arr)

arr=randn(5,3)
    #np.arange(15).reshape(5,3)
print(arr)
#按照第一个维度排序
arr.sort(1)
print(arr)

large_arr = randn(1000)
large_arr.sort()
#5%分位数
print(large_arr[int(0.05*len(large_arr))])

#唯一化以及其他的集合逻辑
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
print(np.unique(names))
ints = np.array([3, 3, 3, 2, 2, 1, 1, 4, 4])
print(np.unique(ints))

#转成set
print(sorted(set(names)))

values = np.array([6, 0, 0, 3, 2, 5, 6])
print(np.in1d(values, [2, 3, 6]))

###线性代数
x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[6., 23.], [-1, 7], [8, 9]])
print(x.dot(y))  # 等价于np.dot(x, y)
np.dot(x, np.ones(3))

np.random.seed(12345)

from numpy.linalg import inv, qr
X = randn(5, 5)
mat = X.T.dot(X)
print(mat)
print(inv(mat))
#矩阵与矩阵的逆相乘得到单位矩阵
print(mat.dot(inv(mat)))
#进行QR分解
q, r = qr(mat)
print(r)
print(q)

###随机数生成
samples = np.random.normal(size=(4, 4))
print(samples)

from random import normalvariate
N = 10
samples = [normalvariate(0, 1) for _ in range(N)]
samples1 = np.random.normal(size=N)

print(samples)
print(samples1)

# 范例：随机漫步
import random
position = 0
walk = [position]
steps = 20
for i in range(steps):
    # print(random.randint(0, 1))
    #随机产生1或者0,如果是1前进一步,如果是0,后退一步
    step = 1 if random.randint(0, 1) else -1
    position += step
    walk.append(position)
#打印出每步的位置
print(walk)

print("========================")

np.random.seed(12345)
nsteps = 20
draws = np.random.randint(0, 2, size=nsteps)
steps = np.where(draws > 0, 1, -1)
print(steps)
walk = steps.cumsum()
print(walk)
print("walk.min()",walk.min())
print("walk.max()",walk.max())
#在什么时候第一次达到2
arg= (np.abs(walk) >= 2).argmax()
print("arg=",arg)
print("========================")

# 一次模拟多个随机漫步
nwalks = 5
nsteps = 4
draws = np.random.randint(0, 2, size=(nwalks, nsteps)) # 0 or 1
steps = np.where(draws > 0, 1, -1)
print(steps)
walks = steps.cumsum(1)
print(walks)

print("walks.min()",walks.min())
print("walks.max()",walks.max())

hits2 = (np.abs(walks) >= 2).any(1)
print("hits2=",hits2)
print(hits2.sum()) # 到达30或-30的数量
print(walks[hits2])
crossing_times = (np.abs(walks[hits2]) >= 2).argmax(1)
print("crossing_times",crossing_times)
print("mean",crossing_times.mean())

print("========================")
steps = np.random.normal(loc=0, scale=0.25, size=(nwalks, nsteps))
print(steps)