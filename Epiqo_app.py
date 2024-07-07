#!/usr/bin/env python
# coding: utf-8

# In[13]:


import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from keras.models import load_model
import requests
from PIL import Image

# Function to check if selected date is a trading day (weekday)
def is_trading_day(date):
    return date.weekday() < 5  # 0=Monday, 4=Friday

# Load your pre-trained models
@st.cache_resource
def load_xgboost_model():
    return joblib.load('xgboost_full_model.pkl')

@st.cache_resource
def load_lstm_model():
    return load_model('lstm_full_model.h5')

@st.cache_resource
def load_svm_model():
    return joblib.load('svm_full_model.pkl')

xgboost_model = load_xgboost_model()
lstm_model = load_lstm_model()
svm_model = load_svm_model()


# Streamlit layout
st.image("https://github.com/Divya-coder-isb/FP2_Epiqo/blob/main/Screenshot%202024-07-03%20135327.png", use_column_width=True)
st.write("### Empowering Informed Investment Decisions Through Epiqo")

st.write("#### Tata Motors Limited (TATAMOTORS.NS)")
st.write("""
It offers its products to fleet owners, transporters, government agencies, defense, public transport utilities, small and medium enterprises (SMEs), agriculture and rural segment, mining and construction industry, etc. The company was incorporated in 1945 and is headquartered in Mumbai, India.
""")

# Date input
selected_date = st.date_input("Choose a date for prediction", min_value=datetime.today())

# Ensure the selected date is a trading day
if not is_trading_day(selected_date):
    st.error("Select the trading days only (Monday to Friday).")
    st.stop()

# Predict button
if st.button('Apply'):
    today = pd.Timestamp.today().normalize()
    difference = (selected_date - today).days

    if difference < 0:
        st.error("Please select a future date.")
    else:
        # Prepare input data format according to each model's requirement
        # This is a placeholder: replace with actual data preparation steps
        input_data = np.array([0])  # Dummy input

        if difference < 90:  # Short term
            prediction = xgboost_model.predict(input_data.reshape(1, -1))
            price_display = f"Short term Adj closing price: {prediction[0]:.2f} INR"
            trend = svm_model.predict(input_data.reshape(1, -1))
        else:  # Long term
            prediction = lstm_model.predict(input_data.reshape(1, 1, -1))  # Adjust input shape for LSTM
            price_display = f"Long term Adj closing price: {prediction[0,0]:.2f} INR"
            trend = svm_model.predict(input_data.reshape(1, -1))

        trend_display = 'Bullish' if trend[0] > 0.5 else 'Bearish' if trend[0] < -0.5 else 'Neutral'
        st.write(price_display)
        
        # Display trend and corresponding image
        if trend_display == 'Bullish':
            st.image("https://github.com/Divya-coder-isb/FP2_Epiqo/blob/main/Bullish.png")
        elif trend_display == 'Bearish':
            st.image("https://github.com/Divya-coder-isb/FP2_Epiqo/blob/main/Bearish.png")
        st.write(f"Market trend prediction: {trend_display}")

