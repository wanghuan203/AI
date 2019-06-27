import numpy as np
import scipy.sparse
import pickle
import xgboost as xgb

from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor


# 加载文件
dtrain = xgb.DMatrix(
    '/input/agaricus.txt.train')
dtest = xgb.DMatrix('input/agaricus.txt.test')

param = {'max_depth':2,'eta':1,'silent':1,'objective':'binary:lgistic'}

watchlist = [(dtest,'eval'),(dtrain,'train')]
num_round = 2

bst = xgb.train(param,dtrain,num_round,watchlist)

#开始预测
preds = bst.predict(dtest)
labels = dtest.get_label()
print('error=%f' % (sum(1 for i in range(len(preds)) if int(preds[i] > 0.5) != labels[i])/float(len(preds))))
bst.save_model('0001.model')


#dump model
bst.dump_model('dump.raw.txt')
#dump model with feature map
bst.dump_model('dump.nice.txt','/input/featmap.txt')

#保存二进制
dtest.save_binary('dtest.buffer')
#保存模型
bst.save_model('xgb.model')


#load model and data in
bst2 = xgb.Booster(model_file='xgb.model')
dtest2 = xgb.DMatrix('dtest.buffer')
preds2 = bst2.predict(dtest2)

#断言，看是否相同
assert np.sum(np.abs(preds2 - preds)) == 0

#二选一，选一个booster
pks = pickle.dumps(bst2)
bst3 = pickle.loads(bst)
preds3 = bst3.predict(dtest2)
#断言，看是否相同
assert np.sum(np.abs(preds3 - preds)) == 0


#创建矩阵
print('开始运行示例,矩阵来自scipy.sparse CSR Matrix')
labels = []
row = []; col = []; dat = []
i = 0
for l in open('input/agaricus.txt.train'):
    arr = l.split()
    labels.append(int(arr[0]))
    for it in arr[1:]:
        k, v = it.split(':')
        row.append(i)
        col.append(int(k))
        dat.append(float(v))
    i += 1
csr = scipy.sparse.csr_matrix((dat,(row,col)))
dtrain = xgb.DMatrix(csr, label = labels)
watchlist = [(dtest,'eval'),(dtrain,'train')]
bst = xgb.train(param,dtrain,num_round,watchlist)


print('开始运行示例，矩阵来自scipy.sparse CSC Matrix')
csc = scipy.sparse.csc_matrix((dat,(row,col)))
dtrain = xgb.DMatrix(csc,label = labels)
watchlist = [(dtest,'eval'),(dtrain,'train')]
bst = xgb.train(param,dtrain,num_round,watchlist)


print('开始运行示例，矩阵来自numpy array')
#numpymat 是numpy array，在内部实现转换成scipy.sparse.csr_matrix，然后转化成DMatrix
npymat = csr.todense()
dtrain = xgb.DMatrix(npymat,label = labels)
watchlist = [(dtest,'eval'),(dtrain,'train')]
bst = xgb.train(param,dtrain,num_round,watchlist)


