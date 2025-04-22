import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Page setup
st.set_page_config(page_title="Stock Anomaly Detection", layout="wide")
st.title("📉 Financial Time-Series Anomaly Detection Tool")

# File uploader
uploaded_file = st.file_uploader("Upload stock CSV file", type=["csv"])

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)  # Handle timezone-aware datetimes
    df['Date'] = df['Date'].dt.tz_convert(None)        # Convert to tz-naive
    df.set_index('Date', inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df

# Financial indicators
def calculate_indicators(df):
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    def rsi(data, window=14):
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    df['RSI_14'] = rsi(df['Close'])
    df['Bollinger_Upper'] = df['SMA_20'] + (df['Close'].rolling(window=20).std() * 2)
    df['Bollinger_Lower'] = df['SMA_20'] - (df['Close'].rolling(window=20).std() * 2)
    return df

# Anomaly detection
def detect_anomalies(df):
    features = df[['Close', 'SMA_20', 'EMA_20', 'RSI_14']].dropna()
    model = IsolationForest(contamination=0.05, random_state=42)
    anomalies = model.fit_predict(features) == -1
    df['Anomaly'] = False
    df.loc[features.index, 'Anomaly'] = anomalies
    return df

# Plotting
def plot_indicators(df, company):
    fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    axs[0].plot(df.index, df['Close'], label='Close Price')
    axs[0].plot(df.index, df['SMA_20'], label='SMA 20', linestyle='--')
    axs[0].plot(df.index, df['EMA_20'], label='EMA 20', linestyle='--')
    axs[0].fill_between(df.index, df['Bollinger_Upper'], df['Bollinger_Lower'], alpha=0.2, label='Bollinger Bands')
    axs[0].set_title(f'{company} - Technical Indicators')
    axs[0].legend()

    axs[1].plot(df.index, df['RSI_14'], color='orange', label='RSI 14')
    axs[1].axhline(70, color='red', linestyle='--')
    axs[1].axhline(30, color='green', linestyle='--')
    axs[1].legend()
    axs[1].set_title('RSI Indicator')

    st.pyplot(fig)

def plot_anomalies(df, company):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df['Close'], label='Close Price')
    if 'Anomaly' in df.columns:
        ax.scatter(df.index[df['Anomaly']], df['Close'][df['Anomaly']], color='red', label='Anomalies', s=30)
    ax.set_title(f'{company} - Price with Anomalies')
    ax.legend()
    st.pyplot(fig)

# Main app
if uploaded_file:
    df = load_data(uploaded_file)
    
    st.sidebar.header("📊 Filter Options")
    companies = df['Company'].unique()
    selected_company = st.sidebar.selectbox("Select Company", sorted(companies))

    filtered_df = df[df['Company'] == selected_company]

    # Date filter
    min_date = filtered_df.index.min().date()
    max_date = filtered_df.index.max().date()
    start_date = st.sidebar.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.sidebar.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)

    filtered_df = filtered_df.loc[(filtered_df.index >= pd.to_datetime(start_date)) &
                                  (filtered_df.index <= pd.to_datetime(end_date))]

    if filtered_df.empty:
        st.warning("No data available for the selected date range.")
    else:
        filtered_df = calculate_indicators(filtered_df)
        filtered_df = detect_anomalies(filtered_df)

        st.subheader("📈 Technical Indicators")
        plot_indicators(filtered_df, selected_company)

        st.subheader("🚨 Detected Anomalies")
        plot_anomalies(filtered_df, selected_company)

        st.subheader("📋 Preview Data")
        st.dataframe(filtered_df.tail(10))

else:
    st.info("Please upload a stock CSV file to begin.")
