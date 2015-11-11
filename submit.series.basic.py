from DataUtils import *
import progressbar as pb
#%%
#test.drop('Store', axis=1, inplace=True)
#test.drop('Date', axis=1, inplace=True)
#testIds = test.Id
#test = addMissingColumnsToTestData(train, test)
#%%
period = 7
skipPeriod = 5
series = getTestSeries_SKIP(train, test, period, skipPeriod, labels)


#%%
#dates = sort(np.unique(test.Date))
#totalSales = np.array([])
#labels = dtest1.feature_names
#for date in dates:
#    series, sales = predictStep(series, testfiltered, date, bstSales.predict, period, labels)
#    totalSales = np.append(totalSales, sales)
#    print date

#%%
dates = sort(np.unique(test.Date))
totalSales = np.array([])
labels = dtest1.feature_names
salesRecord = createSalesRecord(train, test)
#%%
bar = pb.ProgressBar().start()
n = len(dates)
cnt = 0.0
for date in dates:
    series, sales, salesRecord = predictStep_SKIP(series, 
                                testfiltered, 
                                date, 
                                bstSales.predict, 
                                period, 
                                skipPeriod, 
                                salesRecord, 
                                labels)
    totalSales = np.append(totalSales, sales)
    cnt = cnt + 1
    bar.update(cnt*100/n)
    print date       
#%%
submitID = np.load('submitID.npy')
totalSales[totalSales<0] = 0
#testIds = pd.read_csv('test.csv', parse_dates=['Date']).Id.values

data = np.vstack( (submitID, totalSales ) )
np.savetxt('result.csv', data.transpose(), delimiter=',', header='"Id","Sales"', fmt='%.0f', comments='')
