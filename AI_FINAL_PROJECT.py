import yfinance as yf
import pandas as pd
import streamlit as st
from prophet import Prophet
st.title('Stock Market Trading App')

# Define the stock symbols and years
STOCKS = {
    'Apple': 'AAPL',
    'Google': 'GOOGL',
    'Amazon': 'AMZN'
}

YEARS = [1, 2, 3, 4, 5]

# Create the dropdown menus for selecting stock and years
stock = st.selectbox('Select a stock', options=list(STOCKS.keys()))
year = st.selectbox('Select a year', options=YEARS)

# Download the stock data using yfinance
stock_df = yf.download(STOCKS[stock], period=f"{year}y")

# Reset the index and keep only the Date and Close columns
stock_df.reset_index(inplace=True)
stock_df = stock_df[['Date', 'Close']]
stock_df = stock_df.rename(columns={'Date': 'ds', 'Close': 'y'})

# Use Prophet to make a forecast
model = Prophet()
model.fit(stock_df)

future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)
# Allow user to enter the amount they want to invest
investment_amount = st.number_input("Enter the amount you want to invest")

# Calculate the number of shares that can be purchased
last_close_price = stock_df['y'].iloc[-1]
num_shares = int(investment_amount / last_close_price)

# Show the number of shares that can be purchased
st.subheader("Number of shares you can buy")
st.write(num_shares)
# Show the forecast chart
st.title(f"{stock} stock forecast for next {year} years")
st.line_chart(forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']])

# Show the predicted closing price for today
predicted_price = forecast.loc[forecast['ds'] == forecast['ds'].max(), 'yhat'].iloc[0]
st.subheader(f"Predicted {stock} stock price for today: ${predicted_price:.2f}")

# Show the recommendation to buy or sell
last_close_price = stock_df.iloc[-1]['y']
if predicted_price > last_close_price:
    st.success('Recommendation: Buy')
else:
    st.error('Recommendation: Sell')
