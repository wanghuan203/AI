'''
@author: whuan
@contact:
@file: lesson3.py
@time: 2018/10/21 11:09
@desc:numpy数组运算操作
'''

# ndarry是一个多维数组对象
# 实际数据和元数据
# numpy比原生list速度快,效率高,大部分由c写成,开源免费
# Python3 range() 函数返回的是一个可迭代对象（类型是对象），而不是列表类型， 所以打印的时候不会打印列表。
# Python3 list() 函数是对象迭代器，可以把range()返回的可迭代对象转为一个列表，返回的变量类型为列表。
# Python2 range() 函数返回的是列表。

import numpy as np


def pythonsum(n):
    a = list(range(n))
    for i in a:
        print(i)
    b = list(range(n))
    c = []
    for i in range(len(a)):
        a[i] = i ** 2
        b[i] = i ** 3
        c.append(a[i] + b[i])
    return c


print(pythonsum(5))


def numpysum(n):
    a = np.arange(n) ** 2
    b = np.arange(n) ** 3
    c = a + b
    return c


# 效率比较
print(numpysum(5))
a = np.arange(5)
print(a,a.shape)

m = np.array([np.arange(2),np.arange(2)])
print(m)
print(np.zeros(10))
print(np.zeros((2,2,2)))
print(np.empty((2,3)))
empty1= np.empty((2,3))
print(empty1[1,2])
#创建自定义数据类型
t = np.dtype([('name',np.str_,40),('height',np.int32),('weight',np.float32)])
print(t['name'])
item = np.array([('小明',165,65.23),('小红',150,50.01)],dtype=t)
print(item[1]['height'])

#数组切片
a= np.arange(9)
print(a[3:7])
print(a[:7:2])
print(a[::-1])

b=np.arange(24).reshape(4,6)
print(b)

print(a!=1) #判断true Or False 生成对应的数组
print(b.T) #转置
print(b.flatten()) #拉平成为以为数组
print(b.resize((2,12))) #直接修改了b
print(b)

#组合数据
za = np.arange(9).reshape(3,3)
zb = za*3
print(np.hstack((za,zb)))
# print(np.concatenate((za,zb),axis=1))
# print(np.column_stack((za,zb)))
print(np.vstack((za,zb)))
# print(np.concatenate((za,zb),axis=0))
# print(np.row_stack((za,zb)))
print(np.dstack((za,zb)))

#数组分割
print(np.hsplit(za,3))
print(np.vsplit(za,3))
print(np.dsplit(np.arange(27).reshape(3,3,3),3))

print(b.astype(int))



