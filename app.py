import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Streamlit App Title
st.title("📈 Stock Price Predictor App")
st.title("👉 Model Used – CNN + LSTM")

# User Input for Stock Symbol
stock = st.text_input("Enter the Stock Symbol (e.g., GOOG, AAPL, TSLA)", "GOOG")

# Fetch Historical Data from Yahoo Finance
end = datetime.now()
start = datetime(end.year - 20, end.month, end.day)

try:
    google_data = yf.download(stock, start, end)
    if google_data.empty:
        st.error("⚠️ No data found for the given stock symbol. Please check the ticker and try again.")
        st.stop()
except Exception as e:
    st.error(f"⚠️ Error fetching data: {e}")
    st.stop()



# Ensure 'Close' Column Exists
if 'Close' not in google_data.columns:
    st.error("⚠️ The dataset does not contain a 'Close' column. Unable to proceed.")
    st.stop()

# Load Pretrained Model
try:
    model = load_model("Latest_stock_price_model.keras")
except Exception as e:
    st.error(f"⚠️ Error loading model: {e}")
    st.stop()

# Display Stock Data
st.subheader("📊 Stock Data")
st.write(google_data)

# Moving Averages Calculation
google_data['MA_for_250_days'] = google_data['Close'].rolling(250).mean()
google_data['MA_for_200_days'] = google_data['Close'].rolling(200).mean()
google_data['MA_for_100_days'] = google_data['Close'].rolling(100).mean()

# Function to Plot Graphs
def plot_graph(title, *args):
    fig, ax = plt.subplots(figsize=(15, 6))
    labels = ["Close Price"]
    ax.plot(google_data['Close'], label="Close Price", color='blue')

    for idx, (data, color, label) in enumerate(args):
        if data is not None:
            ax.plot(data, color=color, label=label)
            labels.append(label)

    ax.set_title(title)
    ax.legend(labels)
    return fig

# Plot Moving Averages
st.subheader("📈 Original Close Price and MA for 100 days")
st.pyplot(plot_graph("Close Price vs MA (100 Days)", (google_data['MA_for_100_days'], 'red', 'MA for 100 days')))

st.subheader("📈 Original Close Price and MA for 200 days")
st.pyplot(plot_graph("Close Price vs MA (200 Days)", (google_data['MA_for_200_days'], 'green', 'MA for 200 days')))

st.subheader("📈 Original Close Price and MA for 250 days")
st.pyplot(plot_graph("Close Price vs MA (250 Days)", (google_data['MA_for_250_days'], 'orange', 'MA for 250 days')))

st.subheader("📈 Original Close Price vs MA for 100, 200, 250 Days")
st.pyplot(plot_graph("Close Price vs MA (100, 200 & 250 Days)", 
                     (google_data['MA_for_100_days'], 'red', 'MA for 100 days'),
                     (google_data['MA_for_200_days'], 'green', 'MA for 200 days'),
                     (google_data['MA_for_250_days'], 'orange', 'MA for 250 days')))

# Prepare Data for Prediction
splitting_len = int(len(google_data) * 0.7)
x_test = google_data[['Close']].iloc[splitting_len:].copy()

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test)

x_data, y_data = [], []
for i in range(100, len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

# Predict Stock Prices
predictions = model.predict(x_data)

# Inverse Transform to Get Actual Prices
inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)
rmse = np.sqrt(np.mean( (inv_pre - inv_y_test)**2))

# Create DataFrame for Comparison
ploting_data = pd.DataFrame({
    'Original Test Data': inv_y_test.reshape(-1),
    'Predicted Values': inv_pre.reshape(-1)
}, index=google_data.index[splitting_len+100:])

st.subheader("📌 Original vs Predicted Stock Prices")
st.write(ploting_data)

# Plot Predictions vs Actual Values
st.subheader("📊 Original Close Price vs Predicted Close Price")
fig, ax = plt.subplots(figsize=(15,6))
ax.plot(google_data['Close'][:splitting_len+100], label="Data (Not Used)", color='gray')
ax.plot(ploting_data['Original Test Data'], label="Original Test Data", color='blue')
ax.plot(ploting_data['Predicted Values'], label="Predicted Values", color='red')

ax.text(0.05, 0.45, f'RMSE: {rmse:.2f}', transform=ax.transAxes, fontsize=12,
        fontweight='bold', color='black',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='black', alpha=0.8))


ax.legend()
st.pyplot(fig)

st.success("✅ STOCK PRICE PREDICTION COMPLETED SUCCESSFULLY !")
