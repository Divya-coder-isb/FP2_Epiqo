#!/usr/bin/env python
# coding: utf-8

# In[4]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import calendar
from PIL import Image
import requests
import os

# Helper function to download files from GitHub
def download_file(url, filename):
    response = requests.get(url)
    with open(filename, 'wb') as f:
        f.write(response.content)

# Download models from GitHub if they don't exist
if not os.path.exists("xgboost_full_model.pkl"):
    download_file("https://github.com/Divya-coder-isb/FP2_Epiqo/raw/main/xgboost_full_model.pkl", "xgboost_full_model.pkl")
if not os.path.exists("lstm_full_model.h5"):
    download_file("https://github.com/Divya-coder-isb/FP2_Epiqo/raw/main/lstm_full_model.h5", "lstm_full_model.h5")
if not os.path.exists("svm_full_model.pkl"):
    download_file("https://github.com/Divya-coder-isb/FP2_Epiqo/raw/main/svm_full_model.pkl", "svm_full_model.pkl")

# Load models
xgb_model = joblib.load("xgboost_full_model.pkl")
lstm_model = load_model("lstm_full_model.h5")
svm_model = joblib.load("svm_full_model.pkl")

# Download images from GitHub if they don't exist
if not os.path.exists("banner_image.png"):
    download_file("https://github.com/Divya-coder-isb/FP2_Epiqo/raw/main/Screenshot%202024-07-03%20135327.png", "banner_image.png")
if not os.path.exists("bearish.png"):
    download_file("https://github.com/Divya-coder-isb/FP2_Epiqo/raw/main/Bearish.png", "bearish.png")
if not os.path.exists("bullish.png"):
    download_file("https://github.com/Divya-coder-isb/FP2_Epiqo/raw/main/Bullish.png", "bullish.png")

# Load images
banner_image = Image.open("banner_image.png")
bearish_image = Image.open("bearish.png")
bullish_image = Image.open("bullish.png")

# Load scaler for LSTM (Assume you have scaler.pkl saved in your GitHub or adjust accordingly)
if not os.path.exists("scaler.pkl"):
    download_file("https://github.com/Divya-coder-isb/FP2_Epiqo/raw/main/scaler.pkl", "scaler.pkl")
scaler = joblib.load("scaler.pkl")

# Helper functions
def create_lagged_features(data, num_lags=10):
    for i in range(1, num_lags + 1):
        data[f'lag_{i}'] = data['Close'].shift(i)
    data.dropna(inplace=True)
    return data

def predict_xgb(date):
    # Implement XGBoost prediction logic
    return 3900  # Dummy value

def predict_lstm(date):
    # Implement LSTM prediction logic
    return 4000  # Dummy value

def classify_trend(rsi):
    if rsi > 70:
        return 'Bullish', bullish_image
    elif rsi < 30:
        return 'Bearish', bearish_image
    else:
        return 'Neutral', None

def predict_trend(date):
    # Implement SVM trend prediction logic
    rsi = 50  # Dummy RSI value
    return classify_trend(rsi)

def validate_date(selected_date):
    if selected_date.weekday() >= 5:
        st.error("Select the trading days only")
        return False
    return True

# Streamlit UI
st.image(banner_image, use_column_width=True)
st.write("## Tata Motors Limited (TATAMOTORS.NS)")
st.write("It offers its products to fleet owners, transporters, government agencies, defense, public transport utilities, small and medium enterprises (SMEs), agriculture and rural segment, mining and construction industry, etc. The company was incorporated in 1945 and is headquartered in Mumbai, India.")

min_date = datetime.today() + timedelta(days=1)
max_date = min_date + timedelta(days=3650)  # 10 years from today

selected_date = st.date_input("Select a date", min_value=min_date, max_value=max_date, value=min_date)
forecast_type = st.radio("Forecast Type", ('Short term', 'Long term'))

if st.button("Apply"):
    if validate_date(selected_date):
        if forecast_type == 'Short term':
            price = predict_xgb(selected_date)
            st.write(f"Short term Adj closing price: {price} INR")
        else:
            price = predict_lstm(selected_date)
            st.write(f"Long term Adj closing price: {price} INR")

        trend, trend_image = predict_trend(selected_date)
        st.write(f"Market Trend: {trend}")
        if trend_image:
            st.image(trend_image, width=100)

# Display date in the box
st.write(f"Selected Date: {selected_date}")

