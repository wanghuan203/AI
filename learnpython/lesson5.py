'''
@author: whuan
@contact:
@file: lesson5.py
@time: 2018/11/4 20:34
@desc:pandas 学习
series 数组和标签
可以通过标签选取数据,类似索引
dataframe
'''

import sys

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

###pandas
# Series
obj = Series([4, 7, -5, 3])
print(obj)
print(obj.values)
print(obj.index)

obj2 = Series([4, 7, -5, 3], index=['d', 'b', 'a', 'c'])
print(obj2)
print(obj2['a'])
print(obj2[['c', 'a', 'd']])
print(obj.index)
obj2['d'] = 655
print(obj2[['c', 'a', 'd']])
print([obj > 0])
print([obj2 * 2])
print(np.exp(obj2))
print('b' in obj2)
print('e' in obj2)
sdata = {'Ohio': 35000, 'Texas': 71000, 'Oregon': 16000, 'Utah': 5000}
obj3 = Series(sdata)
print(obj3)
states = ['California', 'Ohio', 'Oregon', 'Texas']
obj4 = Series(sdata, index=states)
print(obj4)
print(pd.isnull(obj4))
print(pd.notnull(obj4))
print(obj4.isnull)
print(obj3 + obj4)
obj4.name = 'population'
obj4.index.name = 'state'
print(obj4)

obj.index = ['Bob', 'Steve', 'Jeff', 'Ryan']
print(obj)

print("========================dataframe==========================")

# dataframe
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
frame = DataFrame(data)

print(frame)
DataFrame(data, columns=['year', 'state', 'pop'])
print(frame)

frame2 = DataFrame(data, columns=['year', 'state', 'pop', 'debt'],
                   index=['one', 'two', 'three', 'four', 'five'])
print(frame2)
print(frame2.columns)

print(frame2['state'])
# 两种方式取值
print(frame2.state)
print(frame2.year)
# print(frame2.ix['three'])
print(frame2.loc['three'])
# 默认给所有赋值
frame2['debt'] = 16.5
print(frame2)
# 按照顺序赋值
frame2['debt'] = np.arange(5.)
print(frame2)

# 根据index赋值
val = Series([-1.2, -1.5, -1.7], index=['two', 'four', 'five'])
frame2['debt'] = val
print(frame2)

frame2['eastern'] = frame2.state == 'Ohio'
print(frame2)

# 删除列
del frame2['eastern']
print(frame2.columns)
print(frame2)

pop = {'Nevada': {2001: 2.4, 2002: 2.9},
       'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}
frame3 = DataFrame(pop)
print(frame3)
# 转置
print(frame3.T)

# 有了index,则只显示index的行
print(DataFrame(pop, index=[2001, 2002, 2003]))

pdata = {'Ohio': frame3['Ohio'][:-1],
         'Nevada': frame3['Nevada'][:2]}
print(DataFrame(pdata))
frame3.index.name = 'year';
frame3.columns.name = 'state'
print(frame3)
print(frame3.values)
print(frame2.values)
print("================索引对象============")

obj = Series(range(3), index=['a', 'b', 'c'])
index = obj.index
print(index)

print(index[1:])
# 索引对象不允许修改里面的元素值
# index[1] = 'd'

index = pd.Index(np.arange(3))
print(index)

obj2 = Series([1.5, -2.5, 0], index=index)
# 是否为同一个
print(obj2.index is index)

print(frame3)
print('Ohio' in frame3.columns)
print(2003 in frame3.index)
print(2002 in frame3.index)
'''
print("================数据读取============")

###数据读取
# 读取文本格式数据
df = pd.read_csv('data/ex1.csv')
print(df)

print(pd.read_table('data/ex1.csv', sep=','))

print(pd.read_csv('data/ex2.csv', header=None))
print(pd.read_csv('data/ex2.csv', names=['aa', 'ab', 'ac', 'ad', 'message']))

names = ['af', 'fb', 'cf', 'df', 'message']
print(pd.read_csv('d:data/ex2.csv', names=names, index_col='message'))
print(pd.read_csv('d:data/ex2.csv', names=names))

parsed = pd.read_csv('d:data/csv_mindex.csv', index_col=['key1', 'key2'])
print(parsed)

list(open('d:data/ex3.txt'))
result = pd.read_table('d:data/ex3.txt', sep='\s+')
print(result)
print(pd.read_csv('d:data/ex4.csv', skiprows=[0, 2, 3]))

result = pd.read_csv('d:data/ex5.csv')
print(result)
print(pd.isnull(result))

result = pd.read_csv('d:data/ex5.csv', na_values=['world'])
print(result)

sentinels = {'message': ['foo', 'NA'], 'something': ['two']}
print(pd.read_csv('d:data/ex5.csv', na_values=sentinels))

# 逐行读取文本文件
result = pd.read_csv('d:data/ex6.csv')
print(result)
print(pd.read_csv('d:data/ex6.csv', nrows=5))
chunker = pd.read_csv('d:data/ex6.csv', chunksize=1000)
print(chunker)

tot = Series([])
for piece in chunker:
    print(piece['key'].value_counts())
    tot = tot.add(piece['key'].value_counts(), fill_value=0)

# tot = tot.order(ascending=False)

print(tot)

# 文件写出
data = pd.read_csv('d:data/ex5.csv')
print(data)
data.to_csv('d:data/out.csv')

data.to_csv(sys.stdout, sep='|')

data.to_csv(sys.stdout, na_rep='NULL')

data.to_csv(sys.stdout, index=False, header=False)

data.to_csv(sys.stdout, index=False, columns=['a', 'b', 'c'])

dates = pd.date_range('1/1/2000', periods=7)
print(dates)

ts = Series(np.arange(7), index=dates)
print(ts)
ts.to_csv('tseries.csv')

tser = Series.from_csv('tseries.csv', parse_dates=True)
print(tser)

# 手工处理分隔符格式
import csv

f = open('d:data/ex7.csv')

reader = csv.reader(f)

for line in reader:
    print(line)

lines = list(csv.reader(open('d:data/ex7.csv')))
header, values = lines[0], lines[1:]
data_dict = {h: v for h, v in zip(header, zip(*values))}
print(data_dict)


class my_dialect(csv.Dialect):
    lineterminator = '\n'
    delimiter = ';'
    quotechar = '"'
    quoting = csv.QUOTE_MINIMAL


with open('mydata.csv', 'w') as f:
    writer = csv.writer(f, dialect=my_dialect)
    writer.writerow(('one', 'two', 'three'))
    writer.writerow(('1', '2', '3'))
    writer.writerow(('4', '5', '6'))
    writer.writerow(('7', '8', '9'))
pd.read_table('mydata.csv', sep=';')

print("====================Excel数据=========================")
# Excel数据
# 生成xls工作薄
import xlwt
import xlrd

path = 'd:data/'

wb = xlwt.Workbook()
print(wb)

wb.add_sheet('first_sheet', cell_overwrite_ok=True)
print(wb.get_active_sheet())

ws_1 = wb.get_sheet(0)
print(ws_1)

ws_2 = wb.add_sheet('second_sheet')

data = np.arange(1, 65).reshape((8, 8))
print(data)

ws_1.write(9, 0, 100)
ws_1.write(9, 8, 1)
for c in range(data.shape[0]):
    for r in range(data.shape[1]):
        # print(r, c, data[c, r])
        ws_1.write(r, c, str(data[c, r]))
        ws_2.write(r, c, str(data[r, c]))

wb.save(path + 'workbook.xls')

#生成xlsx工作薄

#从工作薄中读取
book = xlrd.open_workbook(path + 'workbook.xls')
print(book)

book.sheet_names()

sheet_1 = book.sheet_by_name('first_sheet')
sheet_2 = book.sheet_by_index(1)
print(sheet_1)
print(sheet_2.name)

print(sheet_1.ncols, sheet_1.nrows)

cl = sheet_1.cell(0, 0)
print(cl.value)

print(cl.ctype)

print(sheet_2.row(3))

print(sheet_2.col(3))

print(sheet_1.col_values(3, start_rowx=3, end_rowx=7))

print(sheet_1.row_values(3, start_colx=3, end_colx=7))

for c in range(sheet_1.ncols):
    for r in range(sheet_1.nrows):
        print(sheet_1.cell(r, c).value)

#使用pandas读取
xls_file=pd.ExcelFile(path + 'workbook.xls')
table=xls_file.parse('first_sheet')
print(table)

#JSON数据

obj = """
{"name": "Wes",
 "places_lived": ["United States", "Spain", "Germany"],
 "pet": null,
 "siblings": [{"name": "Scott", "age": 25, "pet": "Zuko"},
              {"name": "Katie", "age": 33, "pet": "Cisco"}]
}
"""

import json
result = json.loads(obj)
print(result)

asjson = json.dumps(result)
print(asjson)
siblings = DataFrame(result['siblings'], columns=['name', 'age'])
print(siblings)


#二进制数据格式
#pickle
frame = pd.read_csv('d:data/ex1.csv')
print(frame)
frame.to_pickle('d:data/frame_pickle')

print(pd.read_pickle('d:data/frame_pickle'))

#HDF5格式
store = pd.HDFStore('mydata.h5')
store['obj1'] = frame
store['obj1_col'] = frame['a']
print(store)

print(store['obj1'])

store.close()
# os.remove('mydata.h5')

#使用HTML和Web API
import requests
url = 'https://api.github.com/repos/pydata/pandas/milestones/28/labels'
resp = requests.get(url)
print(resp)

data=json.loads(resp.text)
print(data)
issue_labels = DataFrame(data)
print(issue_labels)
'''

#使用数据库
print("=========使用数据库=========")
import sqlite3

query = """
CREATE TABLE test
(a VARCHAR(20), b VARCHAR(20),
 c REAL,        d INTEGER
);"""

con = sqlite3.connect(':memory:')
con.execute(query)
con.commit()

data = [('Atlanta', 'Georgia', 1.25, 6),
        ('Tallahassee', 'Florida', 2.6, 3),
        ('Sacramento', 'California', 1.7, 5)]
stmt = "INSERT INTO test VALUES(?, ?, ?, ?)"

con.executemany(stmt, data)
con.commit()

cursor = con.execute('select * from test')
rows = cursor.fetchall()
print(rows)
print(cursor.description)
zipd = zip(*cursor.description)
print(list(zipd)[1])
for n in zipd:
    print(n)
print(DataFrame(rows, columns=list(zip(*cursor.description))[0]))

import pandas.io.sql as sql
sql.read_sql('select * from test', con)

