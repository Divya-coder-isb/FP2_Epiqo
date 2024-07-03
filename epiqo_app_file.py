# -*- coding: utf-8 -*-
"""Untitled17.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1cz-M9hVeJsDZSTrx-zr8_jLGccHZamHN
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

# Function to download files from GitHub
def download_file(url):
    response = requests.get(url)
    response.raise_for_status()  # Check that the request was successful
    return response.content

# URLs for models and images
arima_url = 'https://github.com/Divya-coder-isb/FP2_Epiqo/blob/main/arima_model.pkl?raw=true'
svm_url = 'https://github.com/Divya-coder-isb/FP2_Epiqo/blob/main/svm_model.pkl?raw=true'
lstm_url = 'https://github.com/Divya-coder-isb/FP2_Epiqo/blob/main/lstm_model.h5?raw=true'
banner_url = 'https://github.com/Divya-coder-isb/FP2_Epiqo/blob/main/Screenshot%202024-07-03%20135327.png?raw=true'
bearish_url = 'https://github.com/Divya-coder-isb/FP2_Epiqo/blob/main/Bearish.png?raw=true'
bullish_url = 'https://github.com/Divya-coder-isb/FP2_Epiqo/blob/main/Bullish.png?raw=true'

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
        # ARIMA prediction if selected date is less than 3 months from today
        today = datetime.now().date()
        if (selected_date - today) < timedelta(days=90):
            try:
                short_term_price = arima_model.predict([selected_date])[0]
                st.write(f"Short term Adj closing price: {short_term_price} INR")
            except Exception as e:
                st.error(f"Error predicting using ARIMA model: {str(e)}")

        # LSTM prediction if selected date is 3 months or more from today
        else:
            try:
                lstm_model = load_lstm_model()  # Load LSTM model
                long_term_price = lstm_model.predict([selected_date])[0]
                st.write(f"Long term Adj closing price: {long_term_price} INR")
            except Exception as e:
                st.error(f"Error predicting using LSTM model: {str(e)}")

        # SVM prediction for market trend
        try:
            selected_date_str = selected_date.strftime("%Y-%m-%d")
            market_trend = svm_model.predict([selected_date_str])[0]
            if market_trend == 0:
                st.image(bearish_image, caption="Bearish")
            else:
                st.image(bullish_image, caption="Bullish")
        except Exception as e:
            st.error(f"Error predicting using SVM model: {str(e)}")

# Reset button (optional)
#if st.button("Reset"):
#    st.session_state.selected_date = None