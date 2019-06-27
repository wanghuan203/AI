import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import preprocessing
import sklearn as sk
from sklearn.utils import as_float_array
a = np.arange(9)

X =[[1,2,3],[3,5,6]]
Y = np.mean(X, axis=0)

print("np.mean(X)",Y)
print("Y.shape",Y.shape)

XX = np.mat([[10,2,3],[2,3,3],[5,6,3]])
print("XX",XX)
YY = XX.mean(axis=0)
print("np.mean(YY)",YY)
XXX = as_float_array(XX)
print("XXX",XXX)
print("YY.shape",YY.shape)

YY = XXX.mean(axis=0)
print("np.mean(YY)",YY)
print("YY.shape",YY.shape)
print("YY.shape",YY.shape[0])

S = np.array([[10,2,3],[2,3,3],[5,6,3]])
print("S.mean",S.mean(axis=0))

print("S.shape",S.mean(axis=0).shape)


print(XX.shape)
print(XXX.shape)
print(S.shape)

arr = np.arange(1,10,1)
arr = arr.reshape(3,3)
print(arr)

print("=======")

c = np.array([[1, 2, 3, 4],[4, 5, 6, 7], [7, 8, 9, 10]])
print(c.shape) # (3L, 4L)
c.shape=2,-1   #c.reshape((2,-1))
aa = ["dgd,F,1234&dag,F,3434&gdg,F,4878", "gder,F,1234&ere,F,3434&eqre,F,4878", "dgadga,F,1234&dgadogi,F,3434&dgoire,F,4878"]
aa1 = "dgd,F,1234&dag,F,3434&gdg,F,4878"

cc = []

for i in range(len(aa)):
    for j in range(len(aa[i].split("&"))):
        cc.append(("aa", aa[i].split("&")[j]))


print(cc)

n1 = 5
def foo(num):
    n1 = 5
    n1 += num
    return n1


print(foo(n1), n1)

aaaaa="one"

print("dgadgg\dgaggadgag\ngdgdaereq dgadgaaga\\dgrtqerrrr")
a = True
b = False

'''
def map_extract(element):
    return [("a", i) for i in element.split("&") if i]


def map_add(element):
    cc.append(map_extract(element))

#
# for i in range(len(aa)):
#     cc = map_add(map_extract(aa[i]))


bb = map_extract(aa1)
dd = list(map(map_add, aa))
print(cc)
[[('a', 'dgd,F,1234'), ('a', 'dag,F,3434'), ('a', 'gdg,F,4878')],
 [('a', 'gder,F,1234'), ('a', 'ere,F,3434'), ('a', 'eqre,F,4878')],
 [('a', 'dgadga,F,1234'), ('a', 'dgadogi,F,3434'), ('a', 'dgoire,F,4878')]]




#sk.metrics.

#sk.neighbors.NearestNeighbors

print(X)
X1 = sk.preprocessing.normalize(X, norm="l1",axis=0, copy=False)
ss = preprocessing.StandardScaler()
X2 = ss.fit_transform(X)   #默认按列（feature）标准化

# print("======")
print(X1)
print(X2)


for i in a:
    print(i, a[i])
    i = +i

for i, b in enumerate(a):
    print(i, b)

s = 'abcdefg'
print(s[-2:-6:-1])
print(s[-4:-1])

for i in range(1, 100)[6::7]:
    print(i)

for i in range(1, 100):
    if (i % 7==0):
        print(i)

print('=================')
for i in range(7, 100, 7):
    print(i)


# aa = input()
# print(aa)
# del aa
# print(aa)

x=1
y=2
z=1
x,y = y,x
print(float('inf')<float('inf')+1)
print('----------------')

y=1
for x in range(2) :
    while y <3:
        print(y)
        y+=1

        if(y==3):
            break
    else:
        print('aaa')
    print('----')


df4 = pd.DataFrame(np.random.randn(6,6))
df4.describe()
df4.any()



A =[[-1,2,0],[0,3,0],[2,1,-1]]
print("===A",A)
A = np.mat(A)
print("---A",A)
u,sigmoid = np.linalg.eig(A)
print(u,sigmoid)
B = sigmoid * np.mat([[-1,0,0],[0,-1,0],[0,0,3]]) * sigmoid.I
print(B)
'''