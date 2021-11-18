import os

import alpaca_trade_api as tradeapi
import datetime
import pandas as pd
import time
import talib

from main import get_data
import json
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from env.StockEnvPlayerAmmar import StockEnvPlayer


seed = 42
commission = 0

noBacktest = 1

lr = 1e-2
cliprange = 0.3
g = 0.99

ALPACA_API_KEY = "PKS6FOS4JTI3F6GRGAOM"
ALPACA_SECRET_KEY = "zS8y1soTvIvnyCuzsVRz2Mytuf05xvc1HnwTk339"
base_url = "https://paper-api.alpaca.markets"

api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url, api_version="v2")

with open("./config.json", "r") as f:
    config = json.load(f)




def dateparse(x):
    return pd.datetime.strptime(x, "%Y-%m-%d")


def data_gather(symbol, limit=30):  # this function will get the data of one symbol
    try:
        barset = api.get_barset(str(symbol), "day", limit=limit).df
    except:
        return "ticker not found"
    lst = list(barset.index)
    date = [str(i).split(" ")[0] for i in lst]
    price = [float(barset[str(symbol)]["close"][i]) for i in range(limit)]
    ticker = [str(symbol)] * limit

    dct = {"date": date, "ticker": ticker, "adj_close": price}

    return dct


def complete_data_frame(limit=80):
    final_dct = {"date": [], "ticker": [], "adj_close": []}
    stock = [
        "AXP",
        "AAPL",
        "VZ",
        "BA",
        "CAT",
        "JPM",
        "CVX",
        "KO",
        "DIS",
        "XOM",
        "HD",
        "INTC",
        "IBM",
        "JNJ",
        "MCD",
        "MRK",
        "MMM",
        "NKE",
        "PFE",
        "PG",
        "UNH",
        "UTX",
        "WMT",
        "WBA",
        "MSFT",
        "CSCO",
        "GS",
    ]
    lst = []
    for i in range(len(stock)):
        dct = data_gather(stock[i], limit=limit)
        for j in range(len(dct["ticker"])):
            final_dct["date"].append(dct["date"][j])
            final_dct["ticker"].append(dct["ticker"][j])
            final_dct["adj_close"].append(dct["adj_close"][j])
    df = pd.DataFrame(final_dct)
    return df


def add_techicalAnalysis(df):

    close_price = df["adj_close"].values
    #'EMA', 'TEMA',
    #'APO', 'CMO', 'MACD', 'MACD_SIG', 'MACD_HIST', 'MOM', 'PPO', 'ROCP', 'RSI', 'TRIX'
    #'HT_DCPERIOD', 'HT_DCPHASE', 'SINE', 'LEADSINE', 'INPHASE',    'QUADRATURE'

    # =====================================
    # Overlap Studies
    # =====================================
    df["EMA"] = talib.EMA(close_price)
    # TEMA - Triple Exponential Moving Average
    df["TEMA"] = talib.EMA(close_price)

    # =====================================
    # Momentum Indicator Functions
    # =====================================
    # APO - Absolute Price Oscillator
    df["APO"] = talib.APO(close_price, fastperiod=12, slowperiod=26, matype=0)
    # CMO - Chande Momentum Oscillator
    df["CMO"] = talib.CMO(close_price, timeperiod=14)
    # MACD - Moving Average Convergence/Divergence
    df["MACD"], df["MACD_SIG"], df["MACD_HIST"] = talib.MACD(
        close_price, fastperiod=12, slowperiod=26, signalperiod=9
    )
    # MOM - Momentum
    df["MOM"] = talib.MOM(close_price)
    # PPO - Percentage Price Oscillator
    df["PPO"] = talib.PPO(close_price, fastperiod=12, slowperiod=26, matype=0)
    # ROCP - Rate of change Percentage: (price-prevPrice)/prevPrice
    df["ROCP"] = talib.ROCP(close_price, timeperiod=10)
    # RSI - Relative Strength Index
    df["RSI"] = talib.RSI(close_price, timeperiod=14)
    # TRIX - 1-day Rate-Of-Change (ROC) of a Triple Smooth EMA
    df["TRIX"] = talib.TRIX(close_price)

    # =====================================
    # Cycle Indicator Functions
    # =====================================
    # HT_DCPERIOD - Hilbert Transform - Dominant Cycle Period
    df["HT_DCPERIOD"] = talib.HT_DCPERIOD(close_price)
    # HT_DCPHASE - Hilbert Transform - Dominant Cycle Phase
    df["HT_DCPHASE"] = talib.HT_DCPHASE(close_price)
    # HT_SINE - Hilbert Transform - SineWave
    df["SINE"], df["LEADSINE"] = talib.HT_SINE(close_price)
    # HT_TRENDMODE - Hilbert Transform - Trend vs Cycle Mode
    # df['HT_TRENDMODE'] = talib.HT_TRENDMODE(close_price)
    # HT_PHASOR - Hilbert Transform - Phasor Components
    df["INPHASE"], df["QUADRATURE"] = talib.HT_PHASOR(close_price)
    return df


def pre_process(df, addTA="N"):
    df = df.sort_values(
        by=[
            "ticker",
            "date",
        ]
    )
    d = df.date.unique()
    tmp = pd.DataFrame({"date": d}, index=d)

    tickers = df.ticker.unique()
    df2 = pd.DataFrame()
    for t in tickers:
        ticker = df.loc[df.ticker == t]
        # force all stock to have same date range
        ticker = pd.merge(tmp, ticker, how="left", on="date")
        ticker.fillna(method="ffill").fillna(method="bfill")

        # add Techical Analysis to each stock
        if addTA == "Y":
            ticker = add_techicalAnalysis(ticker)
            ticker = ticker.fillna(method="ffill").fillna(method="bfill")

        df2 = pd.concat([df2, ticker], axis=0)
    # df2.to_csv("p3.csv")
    return df2.sort_values(by=["date", "ticker"])


# pre_process(complete_data_frame(),addTA='Y').to_csv('final_data.csv')
# Now we are getting live data now lets do how to get the total cash in alpaca
# df1 = complete_data_frame()

# df.to_csv("test.csv")
account = api.get_account()

# df = pd.read_csv('re.csv',parse_dates=['date'])
#
df = complete_data_frame(limit=100)
df.to_csv("./data/portfolio0.csv")

df = get_data(config, portfolio=0, refreshData=False, addTA="Y")
print("break")
cash = float(account.portfolio_value)
logfile = "./log/"
loop = 0
uniqueId = "ACE"
runtimeId = uniqueId + "_" + str(loop)

title = (
    runtimeId
    + "_Test lr="
    + str(lr)
    + ", cliprange="
    + str(cliprange)
    + ", commission="
    + str(commission)
)

global env
env = DummyVecEnv(
    [
        lambda: StockEnvPlayer(
            df.iloc[-27:],
            logfile + runtimeId + ".csv",
            title,
            seed=seed,
            commission=commission,
            addTA="Y",
            initial_investment=cash,
        )
    ]
)
env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
model = PPO2.load("Dan_RL.pkl")
obs = env.reset()

env.render()

action, _states = model.predict(obs)
obs, rewards, done, info = env.step(action)

#
with open("ace.txt", "r") as f:
    l = f.read()
    a = l.split("\n")


lst_Qty = a[0].split(",")
lst_qnty_int = []
for i in lst_Qty:
    try:
        lst_qnty_int.append(int(i))
    except:
        continue


a[1] = a[1].replace("["," ")
a[1] = a[1].replace("]"," ")
a[1] = a[1].split(" ")

sell = []

for i in a[1]:
    try:
        sell.append(int(i))
    except:
        continue



a[2] = a[2].replace("["," ")
a[2] = a[2].replace("]"," ")
a[2] = a[2].split(" ")

buy = []

for i in a[2]:
    try:
        buy.append(int(i))
    except:
        continue



stock = [
    "AXP",
    "AAPL",
    "VZ",
    "BA",
    "CAT",
    "JPM",
    "CVX",
    "KO",
    "DIS",
    "XOM",
    "HD",
    "INTC",
    "IBM",
    "JNJ",
    "MCD",
    "MRK",
    "MMM",
    "NKE",
    "PFE",
    "PG",
    "UNH",
    "UTX",
    "WMT",
    "WBA",
    "MSFT",
    "CSCO",
    "GS",
]

for m in buy:
    if int(lst_qnty_int[m]) > 0:
        symbol_bars = api.get_barset(stock[m], "minute", 5).df.iloc[0]
        symbol_price = symbol_bars[stock[m]]["close"]
        api.submit_order(
            symbol=stock[m],
            qty=int(lst_qnty_int[m]),
            side="buy",
            type="market",
            time_in_force="gtc",
            order_class="bracket",
            stop_loss={
                "stop_price": symbol_price * 0.99,
                "limit_price": symbol_price * 0.98,
            },
            take_profit={"limit_price": symbol_price * 1.01},
        )
    else:
        continue


for m in sell:
    if int(lst_qnty_int[m]) > 0:
        symbol_bars = api.get_barset(stock[m], "minute", 5).df.iloc[0]
        symbol_price = symbol_bars[stock[m]]["close"]
        api.submit_order(
            symbol=stock[m],
            qty=int(lst_qnty_int[m]),
            side="sell",
            type="market",
            time_in_force="gtc",
            order_class="bracket",
            stop_loss={
                "stop_price": symbol_price * 0.99,
                "limit_price": symbol_price * 0.98,
            },
            take_profit={"limit_price": symbol_price * 1.01},
        )
    else:
        continue

os.remove("ace.txt")
os.remove("./data/portfolio0.csv")


