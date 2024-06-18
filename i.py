# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 01:27:05 2024

@author: PC
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pylab
import scipy.stats as stats
from datetime import datetime
import itertools
from math import sqrt
from pandas.tseries.offsets import DateOffset
from pandas import read_csv
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import pickle as pk
from plotly import graph_objs as go

st.title('Reliance Stock Forecasting')
st.markdown('<style>h1{color: blue;}</style>', unsafe_allow_html=True)

def user_input_features():
    Years = st.number_input('Years of Prediction (max 20 years):', min_value=1, max_value=20, value=1, step=1)
    return Years 

# Load data
data = pd.read_csv("RELIANCE.NS.csv", parse_dates=True, index_col='Date')

# Plot raw data
rw = st.subheader('Plot Closing data')
st.markdown('<style>h3{color: orange;}</style>', unsafe_allow_html=True)

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name="Closing Price"))
    fig.layout.update(title_text='Time Series Data')
    st.plotly_chart(fig)

plot_raw_data()

# Load ARIMA model
try:
    model = pk.load(open('ARIMA.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model file not found. Make sure 'ARIMA.pkl' is in the same directory as your script.")
    st.stop()

# User input for years
years = user_input_features()

# Calculate future dates
last_date = data.index[-1]
future_dates = [last_date + DateOffset(years=i) for i in range(1, years + 1)]
future_data = pd.DataFrame(index=future_dates, columns=['Close'])

# Fit the ARIMA model and make predictions
final_arima = ARIMA(data['Close'], order=(2, 1, 2))
final_arima = final_arima.fit()

# Predict future values
start = len(data)
end = start + years - 1
try:
    forecast = final_arima.predict(start=start, end=end, dynamic=True)
    future_data['Close'] = forecast.values
except Exception as e:
    st.error(f"Error during prediction: {e}")
    st.stop()

st.subheader(f'Forecasting for {years} year(s)')
st.write(future_data)

# Plot Future data
st.subheader('Forecasting plot')
def plot_result_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=future_data.index, y=future_data['Close'], name="Forecasting"))
    fig.layout.update(title_text='Forecasting Data')
    st.plotly_chart(fig)

plot_result_data()


