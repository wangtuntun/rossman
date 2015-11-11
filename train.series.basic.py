import os
os.chdir('/home/farog/rossman')

import pandas as pd
import xgboost as xgb
from utilities import *
from DataUtils import *
from ProfileUtils import *
import h5py

#%%
#train = loadAndProcessTrainData()
#%%
#test = loadAndProcessTestData()
#%%
#train = addMeans(train)
#%%
#testIds = test.Id
#test = addMissingColumnsToTestData(train, test)
#%%
#test = addMeansToTestData(test, train)
#%%

#store = pd.HDFStore('store.h5')
#
#store['train'] = train
#store['test'] = test
#store['testIds'] = testIds
#
#store.close()
#%%
start = time.time()
print('Loading...')
store = pd.HDFStore('store.h5', mode='r')
train = store['train']
test = store['test']
testIds = store['testIds']

end = time.time()
print('Loading done...')
printTime(end-start)

#%%
trainfiltered = filterFeatures(train)
testfiltered = filterFeatures(test)

#%%
trainfiltered = train
testfiltered = test
#%%
trainfiltered.drop(['Customers'], axis=1, inplace=True)
testfiltered.drop(['Customers'], axis=1, inplace=True)

trainfiltered = trainfiltered[trainfiltered.Open]
#testfiltered = testfiltered[testfiltered.Open]
#%%
print('Creating series...')
train1, trainSales, labels = createSeries_SKIP(trainfiltered, 7, 5)
assert (train1.shape[1]==len(labels))
#train1[isnan(train1)] = 0
#%%
#testIds = test.Id
#test = addMissingColumnsToTestData(train, test)
#%%
#trainCustomers = train1[:,0]
#%%
#train2 = np.delete(train1, 0,axis=1)

#%%
#train1=train2
#%%
#print('Shuffling...')
#np.random.seed(6666)
#np.random.shuffle(train1)
#%%
np.random.seed(1305)
msk = np.random.rand(train1.shape[0]) < 0.95
print('Creating DMatrices...')
#%%
dtrain1 = xgb.DMatrix(train1[msk], label = trainSales[msk], missing=NAN, feature_names=labels)
dtest1 = xgb.DMatrix(train1[~msk], label = trainSales[~msk], missing=NAN, feature_names=labels)
#%%
#dtrain1 = xgb.DMatrix('dtrain1')
#dtest1 = xgb.DMatrix('dtest1')
#%%

#dtrain2 = xgb.DMatrix(train2[msk], label = trainCustomers[msk], missing=NAN)
#dtest2 = xgb.DMatrix(train2[~msk], label = trainCustomers[~msk], missing=NAN)
print('Training...')
#%%
param1 = {'booster':'gbtree','max_depth':10, 'eta':0.08, 'silent':1, 'objective':'reg:linear', 'eval_metric':'rmse',
'gamma':0, 'lambda':0, 'alpha':0, 'lambda_bias': 0 }
watchlist1  = [(dtrain1,'train'), (dtest1,'test')]

print(time.ctime())
start = time.time()
num_round1 = 50

bstSales = xgb.train(param1, dtrain1, num_round1, watchlist1, feval=rmspe_metric, early_stopping_rounds=40)

end = time.time()

printTime(end - start)
#%%

bstSales.dump_model('dump.raw.txt')

#
#
#
#
#
#%%
#param2 = {'booster':'gbtree','max_depth':7, 'eta':0.05, 'silent':1, 'objective':'reg:linear', 'eval_metric':'rmse',
#'lambda':0, 'alpha':0, 'lambda_bias': 0 }
#watchlist2  = [(dtrain2,'train'), (dtest2,'test')]
##%%
#start = time.time()
#num_round2 = 200
#
#bstCustomers = xgb.train(param2, dtrain2, num_round2, watchlist2)
#
#end = time.time()
#
#printTime(end - start)