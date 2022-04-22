import pandas as pd
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

data = pd.read_csv('indianinflationdata.csv')
df1=data.T
df1.head()
df1.columns =["cpi"]
df1

df1.index = pd.to_datetime(df1.index)
df1.index.names = ['year']
df1.describe()
upsampled = df1.resample('MS')
final = upsampled.interpolate(method='linear')
final.head()


def time_series():      
    X = final.values
    X = X.astype('float32')
    model = ARIMA(X,order = (2,0,1))
    model_fit = model.fit()

    year = st.sidebar.slider('Forecasted',min_value=2020,max_value=2040)

    n = 2040-year

    forecast = model_fit.forecast(steps = 48-n)
    future_year = [data.index[-1]+i+1 for i in range(0,48-n)]
    future_df = pd.DataFrame(index = future_year,columns = data.columns )
    future_df['Inflation'] = forecast

    df = data.append(future_df)

    st.sidebar.dataframe(future_df)
    st.header("ARIMA model with order = (2,0,1)")
    st.line_chart(df)             


if __name__=='__time_series__':
    time_series()








