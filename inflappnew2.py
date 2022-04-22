import pandas as pd
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

data = pd.read_csv('indianinflationdata.csv')
df1=data.T
df1.columns =["cpi"]
df1.index = pd.to_datetime(df1.index)
df1.index.names = ['year']
upsampled = df1.resample('MS')
final = upsampled.interpolate(method='linear')

st.line_chart(final)
st.area_chart(final)
st.bar_chart(final)

num = st.number_input('Insert the desired minth')
X = final.values
X = X.astype('float32')
model = ARIMA(X, order=(2,0,1))
model_fit = model.fit()
forecast=model_fit.forecast(steps=num)[0]
model_fit.plot_predict(1,769)
forecast






















































