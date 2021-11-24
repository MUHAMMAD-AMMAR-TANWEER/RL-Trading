import numpy as np
import os

import alpaca_trade_api as tradeapi
ALPACA_API_KEY = "PK0BJ4D2A1TK4IUZIX9W"
ALPACA_SECRET_KEY = "k5CjuxYXVukdWUZYouHcGk0HAfmxIl85ShmH0BcE"
base_url = "https://paper-api.alpaca.markets"

api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, base_url, api_version="v2")
account = api.get_account()

print(account)
# with open("ace.txt", "r") as f:
#     l = f.read()
#     a = l.split("\n")
#
#
# lst_Qty = a[0].split(",")
# lst_qnty_int = []
# for i in lst_Qty:
#     try:
#         lst_qnty_int.append(int(i))
#     except:
#         continue
#
#
# a[1] = a[1].replace("["," ")
# a[1] = a[1].replace("]"," ")
# a[1] = a[1].split(" ")
#
# sell = []
#
# for i in a[1]:
#     try:
#         sell.append(int(i))
#     except:
#         continue
#
#
#
# a[2] = a[2].replace("["," ")
# a[2] = a[2].replace("]"," ")
# a[2] = a[2].split(" ")
#
# buy = []



# now the data of prediction done now integrate all
