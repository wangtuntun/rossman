from theano import tensor as T
import theano
import numpy as np
from matplotlib import pylab as plt
import pandas as pd
import operator
import xgboost as xgb
#%%

def printTime(t):    
    if (t > 3600):
        print 'Time needed', t//3600, 'hours', (t%3600//60),'minutes', (t%3600%60//1),'seconds'
    elif (t > 60):
        print 'Time needed', (t//60),'minutes', (t%60//1),'seconds'
    else:
        print 'Time needed', t, 'seconds'

if theano.config.floatX == 'float64':
    epsilon = 1.0e-9
else:
    epsilon = 1.0e-7

def RMSPE(y_true, y_pred):
    return T.sqrt(T.sqrt(T.abs_((y_true - y_pred) / T.clip(T.abs_(y_true), epsilon, np.inf))).mean(axis=-1))

def rmspe(y_true, y_pred):
    return np.mean(rspe(y_true, y_pred))
    
def rspe(y_true, y_pred):
#    msk = y_true!=0
#    y_t = y_true[msk]
#    y_p = y_pred[msk]
#    result = ((y_t-y_p)/y_t)**2
    result = ((y_true-y_pred)/y_true)
    result = result*result
#    return result
    return result[result!=np.inf]
    
def rmspe_metric(preds, dtest):
    return 'rmspe', rmspe(dtest.get_label(), preds)
    
#def rmse_obj(preds, dtest):
#
#%%

def plotFeatureImportance(bst):
    plt.figure(figsize=(10,10), dpi=200)
    xgb.plot_importance(bst, height=0.2)
    plt.gcf().savefig('feature_importance_xgb.png', dpi=200)
    #%%
    
def plotDifference(bst, dtest, target = None):
    #%%
    p = bst.predict(dtest)
    l = 0
    if (target is None):    
        l = dtest.get_label()
    else:
        l = target
    res = np.abs((l-p)/l)
    res = res[~np.isnan(res)]
    res = res[~np.isinf(res)]
    plt.hist(res, range=(0,1), bins=50)
#%%
    
def getScore(bst, dtest, target = None):  
    p = bst.predict(dtest)
    l = 0
    if (target is None):
        l = dtest.get_label()
    else:
        l = target
    res = np.abs((l-p)/l)
    res = res[~np.isnan(res)]
    res = res[~np.isinf(res)]
    res = res*res
    return np.mean(res)
 
def getFScore(bst):
    #%%
    srt = sorted(bst.get_fscore().items(), key=operator.itemgetter(1))
    srt.reverse()
    y = [x[1] for x in srt]    
    srt = [x[0] for x in srt]
    #%%
    fig = plt.figure(figsize=(9,4), dpi=100)
    ax = fig.add_subplot(211)
    x = np.arange(len(srt))  # the x locations for the groups
    width = 0.8
    rects1 = ax.bar(x, y, width=width, bottom=0)
#%%                
    xTickMarks = srt
    ax.set_xticks(x+width)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, rotation=90, fontsize=10)
    
    plt.show()
    
def readFScore():
    return np.load('FSCORE').item()
    