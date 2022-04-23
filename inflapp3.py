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
st.line_chart(final)
st.area_chart(final)

num = st.number_input('Insert the desired number of months',value=0,step=1)
n=num
X= final.values
X= X.astype('float32')
model= ARIMA(X,order = (2,0,1))
model_fit= model.fit()
forecasted= model_fit.forecast(steps = n)
future_year= [final.index[-1]+i+1 for i in range(0,n)]
future_df= pd.DataFrame(index = future_year,columns = final.columns )
future_df['cpi']= forecasted

df= final.append(future_df)

st.sidebar.dataframe(future_df)

st.line_chart(df)             

















































