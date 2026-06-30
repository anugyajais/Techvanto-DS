import yfinance as yf
import streamlit as st
import pandas as pd

# # https://www.makeuseof.com/stock-price-data-using-python/
# ticker = yf.Ticker('GOOGL').info
# market_price = ticker['regularMarketPrice']
# previous_close_price = ticker['regularMarketPreviousClose']
# print('Ticker: GOOGL')
# print('Market Price:', market_price)
# print('Previous Close Price:', previous_close_price)

st.write("""
# Simple Stock Price App

Shown are the stock ***closing price*** and **volume** of Google!

""")

# https://towardsdatascience.com/how-to-get-stock-data-using-python-c0de1df17e75
#define the ticker symbol
tickerSymbol = 'GOOGL'
#get data on this ticker
tickerData = yf.Ticker(tickerSymbol)
#get the historical prices for this ticker
# tickerDf = tickerData.history(period='1d', start='2010-5-31', end='2020-5-31')
tickerDf = tickerData.history(start='2010-5-31', end='2020-5-31')
# Open	High	Low	Close	Volume	Dividends	Stock Splits

st.write("""## Closing Price""")
st.line_chart(tickerDf.Close)
st.write("""## Volume Price""")
st.line_chart(tickerDf.Volume)