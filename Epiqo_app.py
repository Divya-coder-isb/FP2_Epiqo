# -*- coding: utf-8 -*-
"""Untitled64.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1oAod8AG8GBN-PyXr1DBMsErg3LKL_CFv
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import os

# Function to download files from GitHub
def download_file(url, dest):
    if not os.path.exists(dest):
        response = requests.get(url)
        with open(dest, 'wb') as file:
            file.write(response.content)

# URLs of the model files
xgb_url = 'https://github.com/Divya-coder-isb/FP2_Epiqo/raw/main/xgboost_model.pkl'
lstm_url = 'https://github.com/Divya-coder-isb/FP2_Epiqo/raw/main/lstm_model.h5'
lstm_scaler_url = 'https://github.com/Divya-coder-isb/FP2_Epiqo/raw/main/lstm_scaler.pkl'
svm_url = 'https://github.com/Divya-coder-isb/FP2_Epiqo/raw/main/svm_model.pkl'
svm_scaler_url = 'https://github.com/Divya-coder-isb/FP2_Epiqo/raw/main/svm_scaler.pkl'

# URLs of the images
banner_url = 'https://github.com/Divya-coder-isb/FP2_Epiqo/raw/main/Screenshot%202024-07-03%20135327.png'
bearish_url = 'https://github.com/Divya-coder-isb/FP2_Epiqo/raw/main/Bearish.png'
bullish_url = 'https://github.com/Divya-coder-isb/FP2_Epiqo/raw/main/Bullish.png'

# URL of the historical data
data_url = 'https://github.com/Divya-coder-isb/FP2_Epiqo/raw/main/tatamotors_historical_data.csv'

# Download the model files
download_file(xgb_url, 'xgboost_model.pkl')
download_file(lstm_url, 'lstm_model.h5')
download_file(lstm_scaler_url, 'lstm_scaler.pkl')
download_file(svm_url, 'svm_model.pkl')
download_file(svm_scaler_url, 'svm_scaler.pkl')

# Download the images
download_file(banner_url, 'banner.png')
download_file(bearish_url, 'bearish.png')
download_file(bullish_url, 'bullish.png')

# Download the data file
download_file(data_url, 'tatamotors_historical_data.csv')

# Load models and scalers
@st.cache_resource
def load_models():
    xgb_model = joblib.load('xgboost_model.pkl')
    lstm_model = load_model('lstm_model.h5')
    lstm_scaler = joblib.load('lstm_scaler.pkl')
    svm_model = joblib.load('svm_model.pkl')
    svm_scaler = joblib.load('svm_scaler.pkl')
    return xgb_model, lstm_model, lstm_scaler, svm_model, svm_scaler

# Load the historical data
@st.cache_data
def load_data():
    df = pd.read_csv('tatamotors_historical_data.csv')
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d %H:%M:%S%z')
    df.set_index('Date', inplace=True)
    return df

xgb_model, lstm_model, lstm_scaler, svm_model, svm_scaler = load_models()
df = load_data()

# Ensure feature names match
expected_features = xgb_model.feature_names_in_

# Load images
bearish_image = 'bearish.png'
bullish_image = 'bullish.png'
banner_image = 'banner.png'

# Helper functions
def create_features(df):
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week

    # Lag features
    df['lag_1'] = df['Close'].shift(1)
    df['lag_2'] = df['Close'].shift(2)
    df['lag_3'] = df['Close'].shift(3)

    # Rolling features
    df['rolling_mean_3'] = df['Close'].rolling(window=3).mean()
    df['rolling_std_3'] = df['Close'].rolling(window=3).std()

    df['rolling_mean_7'] = df['Close'].rolling(window=7).mean()
    df['rolling_std_7'] = df['Close'].rolling(window=7).std()

    return df

df = create_features(df).dropna()

# Function to create features for a single date
def create_date_features(date, df):
    temp_df = pd.DataFrame({'Date': [date]})
    temp_df.set_index('Date', inplace=True)
    temp_df['dayofweek'] = temp_df.index.dayofweek
    temp_df['quarter'] = temp_df.index.quarter
    temp_df['month'] = temp_df.index.month
    temp_df['year'] = temp_df.index.year
    temp_df['dayofyear'] = temp_df.index.dayofyear
    temp_df['dayofmonth'] = temp_df.index.day
    temp_df['weekofyear'] = temp_df.index.isocalendar().week

    # Concatenate with existing DataFrame to ensure lag and rolling features are calculated
    combined_df = pd.concat([df, temp_df])
    temp_df['lag_1'] = combined_df['Close'].shift(1).iloc[-1]
    temp_df['lag_2'] = combined_df['Close'].shift(2).iloc[-1]
    temp_df['lag_3'] = combined_df['Close'].shift(3).iloc[-1]
    temp_df['rolling_mean_3'] = combined_df['Close'].rolling(window=3).mean().iloc[-1]
    temp_df['rolling_std_3'] = combined_df['Close'].rolling(window=3).std().iloc[-1]
    temp_df['rolling_mean_7'] = combined_df['Close'].rolling(window=7).mean().iloc[-1]
    temp_df['rolling_std_7'] = combined_df['Close'].rolling(window=7).std().iloc[-1]

    temp_df = temp_df[expected_features]
    return temp_df

def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def forecast(date, today):
    if (date - today).days < 90:
        features = create_date_features(date, df)
        st.write("Features for XGBoost model:")
        st.write(features)
        prediction = xgb_model.predict(features)[0]
    else:
        seq_length = 60  # Assuming 60-day sequences for LSTM
        scaled_data = lstm_scaler.transform(df[['Close']])
        seq = scaled_data[-seq_length:]
        prediction = lstm_model.predict(np.array([seq]))
        prediction = lstm_scaler.inverse_transform(prediction)[0, 0]
    return prediction

def classify_trend(date):
    temp_df = pd.DataFrame([{'Date': date}])
    temp_df.set_index('Date', inplace=True)
    temp_df['Close'] = np.nan  # Add a dummy Close value to avoid errors in RSI calculation
    combined_df = pd.concat([df, temp_df])
    rsi = calculate_rsi(combined_df).iloc[-1]
    rsi_scaled = svm_scaler.transform([[rsi]])
    trend = svm_model.predict(rsi_scaled)[0]
    return trend

# Streamlit UI
st.image(banner_image)
st.write("""
# Tata Motors Limited (TATAMOTORS.NS)
It offers its products to fleet owners, transporters, government agencies, defense, public transport utilities, small and medium enterprises (SMEs), agriculture and rural segment, mining and construction industry, etc. The company was incorporated in 1945 and is headquartered in Mumbai, India.
""")

selected_date = st.date_input("Select a date", min_value=datetime.now().date(), max_value=datetime.now().date() + timedelta(days=365))
user_date = pd.to_datetime(selected_date)

if user_date.weekday() >= 5:
    st.error("Select trading days only")
else:
    today = pd.to_datetime('today')

    if st.button('Apply'):
        trend = classify_trend(user_date)

        if (user_date - today).days < 90:
            short_term_price = forecast(user_date, today)
            st.write(f"### Short Term Price: {short_term_price}")
        else:
            long_term_price = forecast(user_date, today)
            st.write(f"### Long Term Price: {long_term_price}")

        if trend == 'bearish':
            st.image(bearish_image, caption='Bearish Trend')
        elif trend == 'bullish':
            st.image(bullish_image, caption='Bullish Trend')
        else:
            st.write("### Neutral")