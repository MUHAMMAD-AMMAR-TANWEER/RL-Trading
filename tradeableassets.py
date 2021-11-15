import alpaca_trade_api as tradeapi
import datetime
import pandas as pd
import time
import talib

ALPACA_API_KEY = "PK4WWYAD0BFG96VR469G"
ALPACA_SECRET_KEY = "txhIqIkkkx7EcYBFqXkVoSuI5cqAZMKGGtvCc796"
base_url = 'https://paper-api.alpaca.markets'

api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url, api_version='v2')

import os
from csv import writer



def data_gather(symbol):# to get the data in right format
    try:
        barset = api.get_barset(str(symbol),"day",limit=2).df
    except:
        return "ticker not found"
    lst= list(barset.index)
    date = str(lst[0]).split(" ")[0]
    price = float(barset[str(symbol)]["close"][0])
    ticker = symbol

    with open('data/portfolio0.csv', 'a', newline='') as f_object:
        writer_object = writer(f_object)
        # Result - a writer object
        # Pass the data in the list as an argument into the writerow() function
        writer_object.writerow([date,ticker,price])
        # Close the file object
        f_object.close()
    return "done"


stock = ["AXP","AAPL","VZ","BA","CAT","JPM","CVX","KO","DIS","XOM","HD","INTC","IBM","JNJ","MCD","MRK","MMM","NKE","PFE","PG","UNH","UTX","WMT","WBA","MSFT","CSCO","GS"]

for i in stock:
    try:
        data_gather(i)
    except:
        print(i)
        continue
    time.sleep(1)
# df = pd.read_csv()





