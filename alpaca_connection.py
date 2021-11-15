import numpy as np
##import yfinance as yf
import json
import alpaca_trade_api as tradeapi

import random
import pandas as pd
import datetime as dt
ALPACA_API_KEY = "PK4WWYAD0BFG96VR469G"
ALPACA_SECRET_KEY = "txhIqIkkkx7EcYBFqXkVoSuI5cqAZMKGGtvCc796"
base_url = 'https://paper-api.alpaca.markets'


df = pd.read_csv("log/ACE_0.csv")

df = df[['asset0_qty', 'asset1_qty', 'asset2_qty', 'asset3_qty', 'asset4_qty',
'asset5_qty', 'asset6_qty', 'asset7_qty', 'asset8_qty', 'asset9_qty',
'asset10_qty', 'asset11_qty', 'asset12_qty', 'asset13_qty',
'asset14_qty', 'asset15_qty', 'asset16_qty', 'asset17_qty',
'asset18_qty', 'asset19_qty', 'asset20_qty', 'asset21_qty',
'asset22_qty', 'asset23_qty', 'asset24_qty', 'asset25_qty',
'asset26_qty', ]]

stock = ["AXP","AAPL","VZ","BA","CAT","JPM","CVX","KO","DIS","XOM","HD","INTC","IBM","'JNJ'","MCD","MRK","MMM","NKE","PFE","PG","UNH","UTX","WMT","WBA","MSFT","CSCO","GS"]
for i, j in df.iterrows():
    alpaca = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url, api_version='v2')
    if i == 1:
        j = list(j)
        for k in range(len(j)):
            if j[k]>=0:
                symbol_bars = alpaca.get_barset(stock[k], 'minute', 5).df.iloc[0]
                symbol_price = symbol_bars[stock[k]]['close']
                alpaca.submit_order(symbol = stock[k],
                qty = 1,
                side = 'buy',
                type = 'market',
                time_in_force = 'gtc',
                order_class = 'bracket',
                stop_loss = {'stop_price': symbol_price * 0.99,
                             'limit_price': symbol_price * 0.98},
                take_profit = {'limit_price': symbol_price * 1.01}
            )


