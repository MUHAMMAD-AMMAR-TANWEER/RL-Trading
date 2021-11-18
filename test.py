import numpy as np
import os

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


# now the data of prediction done now integrate all
