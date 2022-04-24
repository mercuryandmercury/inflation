import pandas as pd
import streamlit as st
import klib
from statsmodels.tsa.arima.model import ARIMA
import ARIMA






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
X = final.values
X = X.astype('float32')
model = ARIMA(X, order=(2,0,1))
model_fit = model.fit()
forecast=model_fit.forecast(steps=60)[0]
model_fit.plot_predict(1,800)
klib.dist_plot(forecast)
st.subheader('predictions for next five years')
st.line_chart(forecast)




































































































