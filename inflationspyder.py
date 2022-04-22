# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 12:35:44 2022

@author: SARKAR
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import klib
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


df=pd.read_csv('indianinflationdata.csv')


df1=df.T
df1.head()
df1.columns =["cpi"]
df1

df1.index = pd.to_datetime(df1.index)
df1.index.names = ['year']
df1.describe()

upsampled = df1.resample('MS')
final = upsampled.interpolate(method='linear')
final.head()
final.to_csv('final.csv', header=True)


from statsmodels.tsa.stattools import adfuller
X=final['cpi'].values
result = adfuller(X)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

if result[0] < result[4]["5%"]:
    print ("Reject Ho - Time Series is Stationary")
else:
    print ("Failed to Reject Ho - Time Series is Non-Stationary")



from statsmodels.tsa.stattools import adfuller
adfuller_test=adfuller(final)
print("pvalue :",adfuller_test[1])

#ACF Plot & PACF Plot
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(final,lags=45)
tsa_plots.plot_pacf(final,lags=45)
plt.show()

final.shape

# separate out a validation dataset
split_point = len(final) - 100
dataset, validation = final[0:split_point], final[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv', header=False)
validation.to_csv('validation.csv', header=False)

print(final.shape)
train=final.iloc[:-100]
test=final.iloc[-100:]
print(train.shape,test.shape)

# Holt method
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing 
from sklearn.metrics import mean_squared_error
hw_model = Holt(train).fit()
pred_hw = hw_model.predict(1,721)
prediction_series = pd.Series(pred_hw,index = final.index)
print(prediction_series)
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
plt.plot(final['cpi'])
plt.plot(prediction_series)
rmse1 = np.sqrt(mean_squared_error(final,pred_hw))
print(rmse1)

#Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
#Exponential Smoothing with Additive trend
exp_add = ExponentialSmoothing(train,trend = 'add').fit()
prediction = exp_add.predict(1,721)
rmse2 = np.sqrt(mean_squared_error(final['cpi'],prediction))
print(rmse2)
prediction_series = pd.Series(prediction,index = final.index)
print(prediction_series)
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
plt.plot(final['cpi'])
plt.plot(prediction_series)

from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
ses_model = SimpleExpSmoothing(train).fit(smoothing_level=0.2)
pred_ses = ses_model.predict(1,721)

prediction_series = pd.Series(pred_ses,index = final.index)
print(prediction_series)
fig, ax = plt.subplots(1, 1, figsize=(15, 5))
plt.plot(final['cpi'])
plt.plot(prediction_series)
rmse3 = np.sqrt(mean_squared_error(final,pred_ses))
print(rmse3)

# evaluate a persistence model
from pandas import read_csv
from sklearn.metrics import mean_squared_error
from math import sqrt
# load data
train = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
# prepare data
X = train.values
X = X.astype('float32')
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]


# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
    yhat = history[-1]
    predictions.append(yhat)
# observation
    obs = test[i]
    history.append(obs)
    print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)


import warnings
from pandas import read_csv
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt


# evaluate an ARIMA model for a given order (p,d,q) and return RMSE
def evaluate_arima_model(X, arima_order):
# prepare training dataset
    X = X.astype('float32')
    train_size = int(len(X) * 0.50)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
# make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
# model_fit = model.fit(disp=0)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
# calculate out of sample error
    rmse = sqrt(mean_squared_error(test, predictions))
    return rmse

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float('inf'), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    rmse = evaluate_arima_model(train, order)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    print('ARIMA%s RMSE=%.3f' % (order,rmse))
                except:
                    continue
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))


# load dataset
train = read_csv('dataset.csv', header=None, index_col=0, parse_dates=True, squeeze=True)
#evaluate parameters
p_values = range(0, 3)
d_values = range(0, 2)
q_values = range(0, 3)
warnings.filterwarnings("ignore")
evaluate_models(train.values, p_values, d_values, q_values)

rmse_df = pd.DataFrame({'Model':['ARIMA','Holt method','ExponentialSmoothing_add','Simple Exponential'],'RMSE':[0.11,rmse1,rmse2,rmse3]})
rmse_df


from pandas import read_csv
from statsmodels.tsa.arima_model import ARIMA
import numpy

# load data
train = read_csv('dataset.csv', header=0, index_col=0, parse_dates=True)
# prepare data
X = train.values
X = X.astype('float32')

# fit model
model = ARIMA(X, order=(2,0,1))
model_fit = model.fit()
forecast=model_fit.forecast(steps=100)[0]
model_fit.plot_predict(1, 721)

#Error on the test data
val=pd.read_csv('validation.csv',header=None)
rmse = np.sqrt(mean_squared_error(val[1], forecast))
rmse

# prepare data
X = final.values
X = X.astype('float32')

# fit model
model = ARIMA(X, order=(2,0,1))
model_fit = model.fit()
forecast=model_fit.forecast(steps=12)[0]
model_fit.plot_predict(1, 769)

forecast=model_fit.forecast(steps=12)[0]
model_fit.plot_predict(1,769)

forecast
 
import pickle 
new = open("infl.pkl", mode = "wb") 
pickle.dump(model, new) 
new.close()


















































































