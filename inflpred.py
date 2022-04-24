import pandas as pd
import streamlit as st
import klib
from statsmodels.tsa.arima.model import ARIMA

data = pd.read_csv('indianinflationdata.csv')
df1=data.T
df1.columns =["cpi"]
df1.index = pd.to_datetime(df1.index)
df1.index.names = ['year']
upsampled = df1.resample('MS')
final = upsampled.interpolate(method='linear')
print(final)
st.subheader('line chart of inflation rate')
st.line_chart(final)
st.subheader('area chart of inflation rate ')
st.area_chart(final)

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

forecast=model_fit.forecast(steps=60)[0]
model_fit.plot_predict(1,800)

forecast
st.area_chart(forecast)

































































































