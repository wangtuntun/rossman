import pandas as pd
import numpy as np
import calendar
from sklearn.preprocessing import MinMaxScaler
import time
from sklearn.externals import joblib
from util import printTime
from utilities import *

from keras.models import model_from_json
import xgboost as xgb
import progressbar as pb
import datetime

#%%
def processDates(store):
    store['Promo2Start'] = store.Promo2SinceYear - 2015
    store['CompetitionStart'] = store.CompetitionOpenSinceYear + (store.CompetitionOpenSinceMonth/12)- 2015
    store['PromoNAN'] = np.isnan(store.CompetitionStart)
    
    del store['Promo2SinceYear']
    del store['CompetitionOpenSinceYear']
    del store['CompetitionOpenSinceMonth']
    
    return store

def processSimpleCategorical(store):    
    labels = ['StoreType','Assortment']
    newLabels = [store]
    for label in labels:
        oneHotEncoding = pd.get_dummies(store[label], prefix=label).astype(np.bool)
        newLabels.append(oneHotEncoding)
    
    store.Promo2 = store.Promo2.astype(np.bool)
    store = pd.concat(newLabels, axis=1)
    return store.drop(labels, axis=1)

def processPromoInterval(store):    
    months = calendar.month_abbr[1:13]
    months[8] = months[8]+'t'  
    month_labels = ['PromoMonth_' + month for month in months]
    month_dict = {months[i]:month_labels[i] for i in range(len(months))}    
    for lab in month_labels:
        store[lab] = False
     
    for index, interval in enumerate(store.PromoInterval):
        if pd.isnull(interval):
            continue
        else:
            labs = interval.split(',')
            for lab in labs:
                store[month_dict[lab]][index] = True
        
    return store.drop('PromoInterval', axis=1)    


def mergeStoreInfo(train, store):
    return pd.merge(train, store, on='Store')
    
    
def extractMonths(train):
    months = [date.month for date in train.Date]
    months_abbr = calendar.month_abbr[1:13]
    month_labels = ['Month_' + month for month in months_abbr]
    
    
    monthsTable = pd.get_dummies(months)
    data = np.zeros((monthsTable.shape[0], 12))
    table = pd.DataFrame(data)
    table.columns = table.columns+1    
    
    table[monthsTable.columns] = monthsTable    
    
    table.columns = month_labels
    
    result = pd.concat([train, table], axis=1)
    result['Month'] = months
    return result
    
        
def extractWeeks(train):
    #weeks = [date.week for date in train.Date]
    # faster version:
    weeks = [date.isocalendar()[1] for date in train.Date]
    train['week'] = weeks
    return train

def loadAndProcessTrainData():
    result = loadAndProcessData(True)
    return reorderColumns(result, ['Sales','Customers','Store'])

def loadAndProcessTestData():
    return loadAndProcessData(False)    

def loadAndProcessData(train):
    start = time.time()
    load = None
    if train:
        load = loadTrainData
    else:
        load = loadTestData
        
    data = load()
    store = pd.read_csv('store.csv')
    
    store = processSimpleCategorical(store)
    store = processPromoInterval(store)
    store = processDates(store)
    #%%
    data = mergeStoreInfo(data, store)
    data = extractMonths(data)
    data = extractWeeks(data)
    
    #rename columns
    renameDict = {
        '0': 'HolidayNone',
        'a': 'HolidayPublic',
        'b': 'HolidayEaster',
        'c': 'HolidayChristmas'
    }
    data.rename(columns = renameDict, inplace=True)
    end = time.time()
    print('Total time:')
    printTime(end - start)    
    
    return data


def loadTestData():
    return loadData('test.csv', False)

def loadTrainData():
    return loadData('train.csv', True)

def loadData(fname, saveSacler = True):
    start = time.time()
    train = pd.read_csv(fname, parse_dates=['Date'])
    train = train.sort(['Store','Date'])
    
    train.Open = train.Open.astype(np.bool, copy=True)
    train.Promo = train.Promo.astype(np.bool, copy=True)
    train.SchoolHoliday = train.SchoolHoliday.astype(np.bool, copy=True)
    train = pd.concat([train, pd.get_dummies(train.StateHoliday).astype(np.bool)], axis = 1)
    
    if (0 in train and '0' in train):
        train['0'] = (train[0] | train['0'])
        del train[0]
    
    train = pd.concat([train, pd.get_dummies(train.DayOfWeek, prefix='DayOfWeek').astype(np.bool)], axis = 1)
#    train = addMeanSalesForWeekDay(train)
#    train = addMeanCustomersForWeekDay(train)
    
    train.drop('StateHoliday', axis=1, inplace=True)
    
#    if saveSacler:
#        scaler = MinMaxScaler()
#        scaler.fit(train.Sales.astype(np.float32))
#        
#        train.Sales = scaler.transform(train.Sales.astype(np.float32))
#        
#        joblib.dump(scaler, 'salesScaler.pkl')
    
    end = time.time()
    print('Loading data:')
    printTime(end-start)

    return train

def addMeans(data):
    start = time.time()
    print('Adding Sales means:')
    data = addMeanSalesForWeekDay(data)
    print 'WeekDay ',
    data = addMeanSalesForMonth(data)
    print 'Month ',  
    data = addMeanSalesForWeekDayOfMonth(data)
    print('WeekOfDayOfMonth ')
    print('Adding Customers means:')    
    data = addMeanCustomersForWeekDay(data)
    print 'WeekDay ',    
    data = addMeanCustomersForMonth(data)
    print 'Month ',  
    data = addMeanCustomersForWeekDayOfMonth(data)
    print('WeekOfDayOfMonth ')
    end = time.time()
    print('Adding means to data:')
    printTime(end-start)
    return data


def addMeanSalesForWeekDayOfMonth(data):
    return data.groupby(['Store','Month', 'DayOfWeek']).apply(addMeanSales_WeekDayOfMonth)
    
def addMeanSales_WeekDayOfMonth(data):
    data['WeekDayOfMonthMeanSales'] = data['Sales'].mean()
    return data

def addMeanSalesForMonth(data):
    return data.groupby(['Store','Month']).apply(addMeanSales_Month)
    
def addMeanSales_Month(data):
    data['MonthMeanSales'] = data['Sales'].mean()
    return data

def addMeanSalesForWeekDay(data):
    return data.groupby(['Store','DayOfWeek']).apply(addMeanSales)

def addMeanSales(data):
    data['DayMeanSales'] = data['Sales'].mean()
    return data

def addMeanCustomersForWeekDay(data):
    return data.groupby(['Store','DayOfWeek']).apply(addMeanCustomers)

def addMeanCustomers(data):
    data['DayMeanCustomers'] = data['Customers'].mean()
    return data

def addMeanCustomersForMonth(data):
    return data.groupby(['Store','Month']).apply(addMeanCustomers_Month)

def addMeanCustomers_Month(data):
    data['MonthMeanCustomers'] = data['Customers'].mean()
    return data

def addMeanCustomersForWeekDayOfMonth(data):
    return data.groupby(['Store','Month','DayOfWeek']).apply(addMeanCustomers_WeekDayOfMonth)

def addMeanCustomers_WeekDayOfMonth(data):
    data['WeekDayOfMonthMeanCustomers'] = data['Customers'].mean()
    return data
    
def getMeansFromTrain(train):
    return train.groupby(['Store','DayOfWeek']).apply(_getFirstRowMeans)
    
def _getFirstRowMeans(df):
    return df.iloc[0][['DayMeanSales',
                        'WeekDayOfMonthMeanSales',
                        'MonthMeanSales',
                        'DayMeanCustomers',
                        'MonthMeanCustomers',
                        'WeekDayOfMonthMeanCustomers'
                        ]]
    
def addMeansToTestData(test, train):
    test['DayMeanSales'] = 0
    test['WeekDayOfMonthMeanSales'] = 0
    test['MonthMeanSales'] = 0
    test['DayMeanCustomers'] = 0
    test['MonthMeanCustomers'] = 0
    test['WeekDayOfMonthMeanCustomers'] = 0
    means = getMeansFromTrain(train)
    for store in np.unique(test.Store):
        for day in range(1,8):
            mean = means.iloc[store, day]
            idx = (test.Store==store) & (test.DayOfWeek==day)
            test.ix[idx, 'DayMeanSales'] = mean.DayMeanSales
            test.ix[idx, 'WeekDayOfMonthMeanSales'] = mean.WeekDayOfMonthMeanSales
            test.ix[idx, 'MonthMeanSales'] = mean.MonthMeanSales
            test.ix[idx, 'DayMeanCustomers'] = mean.DayMeanCustomers
            test.ix[idx, 'MonthMeanCustomers'] = mean.MonthMeanCustomers
            test.ix[idx, 'WeekDayOfMonthMeanCustomers'] = mean.WeekDayOfMonthMeanCustomers
        print('Store: ', store)
    return test
    
def loadModel(modelName):
    start = time.time()
    print('loading model ', modelName, ' ....')
    model = model_from_json(open(modelName + '.json').read())
    model.load_weights(modelName + '.h5')
    end = time.time()
    
    printTime(end-start)
    return model

def reorderColumns(data, labels):
    if (type([]) != type(labels)):
        labels = [labels]
    remainingCols = list(set(data.columns) - set(labels))
    cols = labels + remainingCols
    return data[cols]


def addMissingColumnsToTestData(train, test):
    missingColumns = set(train.columns) - set(test.columns)
    for col in missingColumns:
        test[col] = 0
    test = test[train.columns]
    return test
#####################################################
#####################################################
#####################################################

def createSeries(data, periodLength):
    if (periodLength==1):
        return data.values.astype(np.float32)
    n = len(data)
    m1 = data.shape[1] - 3#Date,Sales.Store corection
    m2 = periodLength
    series1 = np.zeros((n, m1), dtype=np.float32)
    series2 = np.zeros((n, m2), dtype=np.float32)
    start = time.time()
    data.sort(['Store', 'Date'], inplace=True)
    storeIds = np.unique(data.Store)
    target = np.zeros(n )#- (periodLength-1)*len(storeIds))
    cnt = 0
    storenum = 0.0
    numofstores = float(len(storeIds))
    bar = pb.ProgressBar().start()
    labels = []
    for i in storeIds:
        storeRows = getStoreRows(data, i)
        storeSales = storeRows.Sales.values
        storeRows.drop(['Date','Sales','Store'], axis=1, inplace=True)
#        storeCustomers = storeRows.Customers.values
        storeRows = storeRows.values
        for j in xrange(periodLength, len(storeRows)):
            prevSales = storeSales[j-periodLength:j]
#            prevCustomers = storeCustomers[j-periodLength:j]
            features = storeRows[j]#.values
#            prevSales = prevSales.values
            series1[cnt] = features
            series2[cnt] = prevSales
            target[cnt] = storeSales[j]            
            cnt = cnt + 1

        storenum = storenum+1
        
        if (storenum % 100 == 0):
            bar.update(storenum/numofstores*100)
    #%% 
    series = np.concatenate((series1, series2), axis=1)
    #%%
    labels = np.array(data.columns.drop(['Date','Sales','Store']))
    dateLabs = ['Date'+str(i) for i in range(-1,-periodLength-1,-1)]            
    dateLabs.reverse()
    labels = np.append(labels, dateLabs)            
    labels = [str(x).translate(None,'-_') for x in labels]
    end = time.time()
    printTime(end-start)
    return series, target, labels
    
    
def createSeries_SKIP(data, periodLength,skip):
    if (periodLength==1):
        return data.values.astype(np.float32)
    n = len(data)
    m1 = data.shape[1] - 3#Date,Sales.Store corection
    m2 = periodLength
    m3 = skip
    series1 = np.zeros((n, m1), dtype=np.float32)
    series2 = np.zeros((n, m2), dtype=np.float32)
    series3 = np.zeros((n, m3), dtype=np.float32)
    start = time.time()
    data.sort(['Store', 'Date'], inplace=True)
    storeIds = np.unique(data.Store)
    target = np.zeros(n)#- (periodLength-1)*len(storeIds))
    cnt = 0
    storenum = 0.0
    numofstores = float(len(storeIds))
    bar = pb.ProgressBar().start()
    labels = []
    indexStart = max(7*(skip+1), periodLength)
    
    for i in storeIds:
        storeRows = getStoreRows(data, i)
        storeSales = storeRows.Sales.values
        storeRows.drop(['Date','Sales','Store'], axis=1, inplace=True)
#        storeCustomers = storeRows.Customers.values
        storeRows = storeRows.values
        for j in xrange(indexStart, len(storeRows)):
            prevSales = storeSales[j-periodLength:j]
            endSkip = max(j-7*(skip+1), 0)
            prevSkipSales = storeSales[j-7 : endSkip : -7]   
#            print (j-7 , j-7*(skip+1) , -7)
            features = storeRows[j]#.values

            series1[cnt] = features
            series2[cnt] = prevSales
            series3[cnt] = prevSkipSales
            target[cnt] = storeSales[j]            
            cnt = cnt + 1

        storenum = storenum+1
        
        if (storenum % 100 == 0):
            bar.update(storenum/numofstores*100)
    #%% 
    series = np.concatenate((series1, series2, series3), axis=1)
    #%%
    labels = np.array(data.columns.drop(['Date','Sales','Store']))
#%%    
    dateLabs = ['Date'+str(i) for i in range(-1,-periodLength-1,-1)]            
    dateSkipLabs = ['DateSkip'+str(i) for i in range(7,(skip+1)*7,7)]
    dateLabs.reverse()
    dateSkipLabs.reverse()
#%%
    labels = np.append(labels, dateSkipLabs)
    #%%
    labels = np.append(labels, dateLabs)
    labels = [str(x).translate(None,'-_') for x in labels]
#%%
    end = time.time()
    printTime(end-start)
    return series, target, labels
   
def getStoreRows(data, s):
    d=data.Store.values
    a=np.searchsorted(d, s)
    b=np.searchsorted(d, s, side='right')
    return data.iloc[a:b]
    
    
def getTestSeries(data, future, periodLength, labels):
#    data = filterFeatures(data)
#    future = filterFeatures(future)
    data.sort(['Store', 'Date'], inplace=True)
    future.sort(['Date','Store'], inplace=True)
    storeIds = np.unique(future.Store)
#%%    
    minDate = min(future.Date)
    idx = np.where(future.Date == minDate)
    firstDay = future.iloc[idx]
#    firstDay = firstDay.drop(['Sales','Date'], axis = 1)
        
    firstDay.columns = [x.translate(None,'_') for x in firstDay.columns]
    lab = [x for x in firstDay.columns if labels.__contains__(x)]
    lab.insert(0, 'Store')
    firstDay = firstDay[lab]
   #%%     
    m = data.shape[1] + (periodLength) - 3 #Date,Sales,Customers corection
    
    n = len(np.unique(data.Store))
    result = np.zeros((n, m))
    cnt = 0
    
    result = np.zeros((len(storeIds), m))
    for storeId in storeIds:
        storeRows = getStoreRows(data, storeId)
#        print('--',storeId)
        storeIdx = np.where(firstDay.Store == storeId)
#        print('### ',storeIdx)
#        print firstDay.columns
        features = firstDay.iloc[storeIdx].drop('Store', axis=1).values.squeeze()
        prevSales = storeRows.Sales[-periodLength:]
#        prevCustomers = storeRows.Customers[-periodLength:]
        item = np.hstack((features, prevSales))
#        print prevSales.shape, features.shape
        result[cnt] = item
        cnt = cnt + 1
        if (cnt%100==0):        
            print(cnt)
    return result
        
        
def getTestSeries_SKIP(data, future, periodLength, skip, labels):
#    data = filterFeatures(data)
#    future = filterFeatures(future)
    data.sort(['Store', 'Date'], inplace=True)
    future.sort(['Date','Store'], inplace=True)
    storeIds = np.unique(future.Store)
#%%    
    minDate = min(future.Date)
    idx = np.where(future.Date == minDate)
    firstDay = future.iloc[idx]
#    firstDay = firstDay.drop(['Sales','Date'], axis = 1)
        
    firstDay.columns = [x.translate(None,'_') for x in firstDay.columns]
    lab = [x for x in firstDay.columns if labels.__contains__(x)]
    lab.insert(0, 'Store')
    firstDay = firstDay[lab]
   #%%     
    m = data.shape[1] + (periodLength) + skip - 3 #Date,Sales,Customers corection
    
    n = len(np.unique(data.Store))
    result = np.zeros((n, m))
    cnt = 0
    
    result = np.zeros((len(storeIds), m))
    for storeId in storeIds:
        storeRows = getStoreRows(data, storeId)
#        print('--',storeId)
        storeIdx = np.where(firstDay.Store == storeId)
#        print('### ',storeIdx)
#        print firstDay.columns
        features = firstDay.iloc[storeIdx].drop('Store', axis=1).values.squeeze()
        prevSales = storeRows.Sales[-periodLength:]
        prevSkipSales = storeRows.Sales[-7:-7*(skip+1):-7]
#        prevCustomers = storeRows.Customers[-periodLength:]
        item = np.hstack((features, prevSkipSales, prevSales))
#        print prevSales.shape, features.shape
        result[cnt] = item
        cnt = cnt + 1
        if (cnt%100==0):        
            print(cnt)
    return result

def getEarliestDate(data, periodLength):
    #%%
    minStoreId = min(data.Store)
    idx = np.where(data.Store == minStoreId)
    t = data.iloc[idx]
    t.sort(['Date'], inplace=True)
    #%%
    return t.Date.iloc[ -periodLength + 1 ]
    
def getLastSales(data, periodLength):
#%%    
    minDate = getEarliestDate(data, periodLength)
    m = len(np.unique(data.Store))
    result = np.zeros((m, periodLength - 1))
    cnt = 0    
   #%% 
    
    for store in np.unique(data.Store):
        storeRows = getStoreRows(data, store)
        idx = np.where(storeRows.Date >= minDate)
        result[cnt] = storeRows.Sales.values[idx]
        
        cnt = cnt + 1
#        print(cnt)
    #%%
    return result
    
def predictStep(series, future, date, predictSalesFunc, periodLength, labels):    
    
    idx = np.where(future.Date == date)
    currDay = future.iloc[idx]
    currDay.sort(['Store'], inplace = True)
    currDay.drop(['Date', 'Sales','Store'], axis=1, inplace=True)
    series[:,:-periodLength] = currDay.values
    
    #prediction
#    customers = predictCustFunc(xgb.DMatrix(series, missing=np.nan))
#    series = np.insert(series, 0, customers, axis = 1)
    sales = predictSalesFunc(xgb.DMatrix(series, missing=np.nan, feature_names = labels))
    
    #prepare for next round
    series[:,-periodLength:-1] = series[:,-periodLength+1:]
    series[:,-1] = sales
    
#    series[:,-2*periodLength:-periodLength-1] = series[:,-2*periodLength+1:-periodLength]
#    series[:,-periodLength-1] = customers
#    series = np.delete(series, 0, axis = 1)
    return series, sales
   

 
def predictStep_SKIP(series, future, date, predictSalesFunc, periodLength, skip, salesRecord, labels):
#%%    
    tdelta = datetime.timedelta(days=7)
   #%% 
    
    
    idx = np.where(future.Date == date)
    currDay = future.iloc[idx]
    currDay.sort(['Store'], inplace = True)
    currDay.drop(['Date', 'Sales','Store'], axis=1, inplace=True)
    assert series.shape[1] == currDay.shape[1] + periodLength + skip    
    series[:,:-periodLength-skip] = currDay.values
    
    #prediction
    sales = predictSalesFunc(xgb.DMatrix(series, missing=np.nan, feature_names = labels))
    
    dateIdx = salesRecord.Date == date
    assert sum(dateIdx) == len(sales)    
    salesRecord.ix[dateIdx,'Sales'] = sales
    
    #prepare for next round
    series[:,-periodLength:-1] = series[:,-periodLength+1:]
    series[:,-1] = sales
#%%
    n = series.shape[0]
    
    skipDates = np.zeros((n,1))
    for i in range(1,skip+1):
        td = date - i*tdelta
        skipSales = salesRecord[salesRecord['Date'] == td].Sales.values
        skipSales = skipSales.reshape((skipSales.shape[0],1))
#        print skipDates.shape, skipSales.shape
        skipDates = np.hstack(( skipSales, skipDates))
    skipDates = np.delete(skipDates, -1, axis=1)
#%%
#    salesRecord[salesRecord['Date'].isin(skipDates)]

    series[:,-periodLength-skip:-periodLength] = skipDates

    return series, sales, salesRecord
        
    #%%    
def createSalesRecord(train, test):
    trainSub = train[['Store','Sales','Date']]
    testSub = pd.DataFrame(test[['Store','Date']])
    result = pd.merge(trainSub, testSub, how='outer')
    stores = np.unique(test.Store)
    result = result[result['Store'].isin(stores)]
    return result
    #%%
##########################################################
#%%    
    
def filterFeatures(data):
    labs = []
    labs.extend( getColumnsStartingWith(data.columns, 'Assortment') )
    labs.extend( getColumnsStartingWith(data.columns, 'Holiday') )
#    labs.extend( getColumnsStartingWith(data.columns, 'Month') )
#    labs.extend( getColumnsStartingWith(data.columns, 'PromoMonth') )
    labs.extend( getColumnsWith(data.columns, 'Mean') )
    labs.extend(['Sales','Customers','Store','Date'])
    labs.extend(['DayOfWeek', 'week', 'Promo', 'Open', 'CompetitionDistance'])
    return reorderColumns(data[labs], ['Sales','Customers','Store'])    
    
#%%

def getColumnsStartingWith(columns, label):
    return [x for x in columns if (type(x)==type('str') and x.startswith(label))]
    
def getColumnsWith(columns, label):
    return [x for x in columns if (type(x)==type('str') and x.find(label)!=-1)]