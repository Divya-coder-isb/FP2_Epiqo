# -*- coding: utf-8 -*-
"""Untitled25.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Z9idct4qKPaPMhCimQzOYnv2yr_Rzrfo
"""

import streamlit as st
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from PIL import Image
import holidays
import requests
import io
import tempfile
import os
from sklearn.preprocessing import MinMaxScaler

# Function to download files from GitHub
def download_file(url):
    response = requests.get(url)
    response.raise_for_status()  # Check that the request was successful
    return response.content

# URLs for models and images
arima_url = 'https://github.com/Divya-coder-isb/FP2_Epiqo/blob/main/arima_model.pkl?raw=true'
svm_url = 'https://github.com/Divya-coder-isb/FP2_Epiqo/blob/main/svm_model.joblib?raw=true'
lstm_url = 'https://github.com/Divya-coder-isb/FP2_Epiqo/blob/main/lstm_model.h5?raw=true'
banner_url = 'https://github.com/Divya-coder-isb/FP2_Epiqo/blob/main/Screenshot%202024-07-03%20135327.png?raw=true'
bearish_url = 'https://github.com/Divya-coder-isb/FP2_Epiqo/blob/main/Bearish.png?raw=true'
bullish_url = 'https://github.com/Divya-coder-isb/FP2_Epiqo/blob/main/Bullish.png?raw=true'
historical_data_url = 'https://github.com/Divya-coder-isb/FP2_Epiqo/blob/main/tatamotors_stock_data.csv?raw=true'

# Load ARIMA model
try:
    arima_model = joblib.load(io.BytesIO(download_file(arima_url)))
except (ValueError, requests.exceptions.RequestException) as e:
    st.error(f"Error loading ARIMA model: {str(e)}")
    st.stop()

# Load SVM model
try:
    svm_model = joblib.load(io.BytesIO(download_file(svm_url)))
except (ValueError, requests.exceptions.RequestException) as e:
    st.error(f"Error loading SVM model: {str(e)}")
    st.stop()

# Function to load LSTM model from a temporary file
def load_lstm_model():
    lstm_model_bytes = download_file(lstm_url)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        tmp.write(lstm_model_bytes)
        tmp.flush()
        lstm_model_path = tmp.name

    try:
        lstm_model = tf.keras.models.load_model(lstm_model_path)
    finally:
        # Clean up the temporary file
        os.remove(lstm_model_path)

    return lstm_model

# Load images
banner_image = Image.open(io.BytesIO(download_file(banner_url)))
bearish_image = Image.open(io.BytesIO(download_file(bearish_url)))
bullish_image = Image.open(io.BytesIO(download_file(bullish_url)))

# Set holiday list (India)
indian_holidays = holidays.IN()

# Helper function to check if a date is a trading day
def is_trading_day(date):
    return date.weekday() < 5 and date not in indian_holidays

# Fetch historical data
def fetch_historical_data():
    data = pd.read_csv(io.BytesIO(download_file(historical_data_url)))
    data['Date'] = pd.to_datetime(data['Date'])
    return data

# Prepare data for LSTM
def prepare_lstm_data(data):
    data = data[['Adj Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

# Prepare data for SVM
def prepare_svm_data(data):
    data['Returns'] = data['Adj Close'].pct_change()
    data = data.dropna()
    X = data[['Open', 'High', 'Low', 'Close', 'Volume', 'Returns']]
    return X

# Streamlit UI
st.image(banner_image, width=900, use_column_width=False)
st.title("Empowering Informed Investment Decisions Through Epiqo")

st.markdown("""
**Tata Motors Limited (TATAMOTORS.NS)**
It offers its products to fleet owners, transporters, government agencies, defense, public transport utilities, small and medium enterprises (SMEs), agriculture and rural segment, mining and construction industry, etc. The company was incorporated in 1945 and is headquartered in Mumbai, India.
""")

# Input field - Period
selected_date = st.date_input("Period", key="selected_date")

# Validate selected date
if selected_date and not is_trading_day(selected_date):
    st.error("Select trading days only")
    st.stop()

# Apply button
if st.button("Apply"):
    if selected_date:
        # Fetch historical data
        data = fetch_historical_data()

        # ARIMA prediction if selected date is less than 3 months from today
        today = datetime.now().date()
        if (selected_date - today) < timedelta(days=90):
            try:
                short_term_price = arima_model.forecast(steps=1)[0]
                st.write(f"Short term Adj closing price: {short_term_price} INR")
            except Exception as e:
                st.error(f"Error predicting using ARIMA model: {str(e)}")

        # LSTM prediction if selected date is 3 months or more from today
        else:
            try:
                lstm_model = load_lstm_model()  # Load LSTM model
                X, y, scaler = prepare_lstm_data(data)
                lstm_input = X[-1].reshape(1, 60, 1)
                long_term_price = lstm_model.predict(lstm_input)[0][0]
                long_term_price = scaler.inverse_transform([[long_term_price]])[0][0]
                st.write(f"Long term Adj closing price: {long_term_price} INR")
            except Exception as e:
                st.error(f"Error predicting using LSTM model: {str(e)}")

        # SVM prediction for market trend
        try:
            X_svm = prepare_svm_data(data)
            svm_input = X_svm.iloc[-1].values.reshape(1, -1)
            market_trend = svm_model.predict(svm_input)[0]
            if market_trend == 0:
                st.image(bearish_image, caption="Bearish")
            else:
                st.image(bullish_image, caption="Bullish")
        except Exception as e:
            st.error(f"Error predicting using SVM model: {str(e)}")