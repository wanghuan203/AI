'''
@author: whuan
@contact:
@file: lesson1.py
@time: 2018/10/20 16:53
@desc:基本数据类型
'''

print("hello world")

if False:
    print("true")
else:
    print("false")
    print("aa")
days = ["monday", "tuesday",
        "friday"]
print(days)
''' aaa
'''
# aaaa
word = """aaaa
bbb
ccc
ddd
"""
print(word)

# 需要用户输入才能继续运行
# input("\n\npress the enter key to exit.")
import sys;

x = 'foo';
sys.stdout.write(x + '\n')

# 帮助
help(sys.stdout.write)

# 变量
counter = 100  # 整型
miles = 1000.0  # 浮点型
name = "小明"
# 所有的变量使用之前都需要赋值
# 变量赋值的操作既是声明也是定义
print(counter)
print(miles)
print(name)

# 可以给多个变量赋值
a = b = c = 1
print(a, b, c)
a, b, c = 1, 2, "字符"
print(a, b, c)

'''
数值型
int有符号的整型
long长整型
float浮点型
complex复数
'''
# 删除操作
# del a
# print(a)

# 字符串
s = 'Ilovepython'
print(s[1:5])
print(s[5:-1])
print(s * 2)
print(s + " and you?")

# 列表 []
list = ['adbc', 343, 2, 343, 'john', 30.11]
tinylist = ['dgd', 124]
print(list)
print(list[2])
print(tinylist * 2)

# 元祖 tuple 元素不可修改
tuple = ()

# 字典
dict = {}
dict['one'] = "this is one"
dict[2] = "this is two"

print(dict.keys())
print(dict.values())
print(dict[2])
print(1!=2)
