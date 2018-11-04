'''
@author: whuan
@contact:
@file: lesson2.py
@time: 2018/10/20 19:47
@desc:控制语句,文件操作
'''

# 条件语句
import math

import time

num = 5
if num == 3:
    print("3")
elif num == 5:
    print("5")
else:
    print("no")

# 循环
a = 1
while (a <= 10):
    print(a)
    a += 1
else:
    print("结束")

for letter in "Ilovepython":
    print(letter)
else:
    print("over")

fruits = ['banana', 'apple', 'mango']
for fruit in fruits:
    print(fruit)

for index in range(len(fruits)):
    print("当前水果", fruits[index])
else:
    print("没有水果了")

print(math.e)
print(math.pi)

print(time.time())
print(time.asctime(time.localtime(time.time())))


def printInfo(name="小花", age=10, *vartuple):
    print("名字:", name, ",年龄:", age)
    for var in vartuple:
        print("朋友是:", var)
    return


printInfo("小明", 10, '小黑', '小绿')
# 使用了不定长函数后不能再指定参数顺序
# printInfo(age=18,name="小红","小黑","小路")
printInfo()

'''
lambda匿名函数
'''
sum = lambda a, b: a + b
sum(1, 8)

print(sum(1, 10))

'''
open文件
读文件
#r表示是文本文件，rb是二进制文件
写文件
#w表示是文本文件，wb是二进制文件
'r'：读
'w'：写
'a'：追加
'r+' == r+w（可读可写，文件若不存在就报错(IOError)）
'w+' == w+r（可读可写，文件若不存在就创建）
'a+' ==a+r（可追加可写，文件若不存在就创建）
'''
try:
    file = open("foo.txt", "r")
    str = file.read()
    print(str)
    print("文件名:", file.name)
except ValueError:
    print("文件操作错误",IOError)
else:
    print("文件操作成功")
finally:
    print("结束了")
    file.close()

try:
    file = open("foo.txt", "a")
    str = file.write("\npython is very intersting")
    print(str)
    print("文件名:", file.name)
except ValueError:
    print("文件操作错误",IOError)
else:
    print("文件操作成功")
finally:
    print("结束了")
# aa="these are words my written"
# file.write(aa)
# file.write("these are words my written".decode('utf8'))
# file.write(aa.split(','.encode(encoding="utf-8")))
file.close()
