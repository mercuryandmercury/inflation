import pandas as pd
import pickle
import pandas as pd
import streamlit as st
import klib
from statsmodels.tsa.arima.model import ARIMA


model=pickle.load(open('infl2.pkl','rb'))

print(model)

data = pd.read_csv('indianinflationdata.csv')
df1=data.T
df1.columns =["cpi"]
df1.index = pd.to_datetime(df1.index)
df1.index.names = ['year']
upsampled = df1.resample('MS')
final = upsampled.interpolate(method='linear')
print(final)
st.line_chart(final)
st.area_chart(final)

st.line_chart(model)





























































































