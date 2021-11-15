import pandas as pd
from main import get_data
import json
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from env.StockEnvPlayer import StockEnvPlayer

with open('./config.json', 'r') as f:
    config = json.load(f)

df = get_data(config, portfolio=0, refreshData=False, addTA="Y")

seed = 42
commission = 0

noBacktest = 1

lr = 1e-2
cliprange = 0.3
g = 0.99


def evaluate(model, num_steps=1000):
    episode_rewards = [0.0]
    obs = env.reset()




    env.render()

    for i in range(num_steps):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()

        # Stats
        episode_rewards[-1] += rewards
        if done:
            obs = env.reset()
            episode_rewards.append(0.0)

    return np.sum(episode_rewards)

before = np.zeros(noBacktest)

after = np.zeros(noBacktest)
backtest = np.zeros(noBacktest)
train_dates = np.empty(noBacktest, dtype="datetime64[s]")
start_test_dates = np.empty(noBacktest, dtype="datetime64[s]")#date 1970
end_test_dates = np.empty(noBacktest, dtype="datetime64[s]")

dates = np.unique(df.date) #return the list of dates
logfile = "./log/"

cutoff_date = np.datetime64('2018-01-04')
    # backtest=1 uses cut of date to split train/test
cutoff_date = np.datetime64(cutoff_date)#cutoff_date is between training and testing I set to 2018-01-04
# a = np.where(dates < cutoff_date)[0]
# print(a)
if backtest == 1:
    a = np.where(dates < cutoff_date)[0]
    b = np.where(dates >= cutoff_date)[0]
    s = []
    s.append((a, b))
else:
    a = np.where(dates < cutoff_date)[0]
    b = np.where(dates >= cutoff_date)[0]
    s = []
    s.append((a, b))
loop = 0
uniqueId = "ACE"
for train_date_index, test_date_index in s:#loop only runs once
    train = df[df.date.isin(dates[train_date_index])]
    test = df[df.date.isin(dates[test_date_index])]

    runtimeId = uniqueId + "_" + str(loop)
    train_dates[loop] = max(train.date)
    start_test_dates[loop] = min(test.date)#2018
    end_test_dates[loop] = max(test.date)#2021

    # model = PPO2.load("Dan_RL.pkl")
    # print(test.head())

    last_test = test.iloc[-108:-54]
    # print(last_test.head(27))

    title = runtimeId + "_Test lr=" + \
            str(lr) + ", cliprange=" + str(cliprange) + ", commission=" + str(commission)

    global env
    env = DummyVecEnv(
        [lambda: StockEnvPlayer(last_test, logfile + runtimeId + ".csv", title, seed=seed, commission=commission,
                                addTA="Y")])
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)


    steps = len(np.unique(last_test.date))
    print((steps))

    model = PPO2.load("Dan_RL.pkl")
    episode_rewards = [0.0]
    obs = env.reset()
    # #
    # #
    # #
    # #
    # # env.render()
    # # action, _states = model.predict(obs)
    env.render()
    for i in range(1):
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        print(obs)


        # env.render()
        # print(buy_signal)
        # print("++++++++++++++")
        # print(sell_signal)
        # print("++++++++++++++++")




    #
    #
    #
    #
    #
    #
    #     print(buy_index)
    #     print(sell_index)
    #     print("************")
    #     print(action)#now with action comes we will be using step to drag model
    #     print("**************")
    #     print(_states)#it is none
    # backtest[loop] = evaluate(model, num_steps=steps)



    # a = evaluate(model, num_steps=steps)




def bot(x):
    before = np.zeros(noBacktest)

    after = np.zeros(noBacktest)
    backtest = np.zeros(noBacktest)
    train_dates = np.empty(noBacktest, dtype="datetime64[s]")
    start_test_dates = np.empty(noBacktest, dtype="datetime64[s]")  # date 1970
    end_test_dates = np.empty(noBacktest, dtype="datetime64[s]")

    dates = np.unique(df.date)  # return the list of dates
    logfile = "./log/"

    cutoff_date = np.datetime64('2018-01-04')
    # backtest=1 uses cut of date to split train/test
    cutoff_date = np.datetime64(cutoff_date)  # cutoff_date is between training and testing I set to 2018-01-04
    if backtest == 1:
        a = np.where(dates < cutoff_date)[0]
        b = np.where(dates >= cutoff_date)[0]
        s = []
        s.append((a, b))
    else:
        a = np.where(dates < cutoff_date)[0]
        b = np.where(dates >= cutoff_date)[0]
        s = []
        s.append((a, b))
    loop = 0
    uniqueId = "ACE"
    for train_date_index, test_date_index in s:  # loop only runs once
        train = df[df.date.isin(dates[train_date_index])]
        test = df[df.date.isin(dates[test_date_index])]

        runtimeId = uniqueId + "_" + str(loop)
        train_dates[loop] = max(train.date)
        start_test_dates[loop] = min(test.date)  # 2018
        end_test_dates[loop] = max(test.date)  # 2021

        # model = PPO2.load("Dan_RL.pkl")
        # print(test.head())

        last_test = test.iloc[-17*int(x):]

        title = runtimeId + "_Test lr=" + \
                str(lr) + ", cliprange=" + str(cliprange) + ", commission=" + str(commission)

        global env
        env = DummyVecEnv(
            [lambda: StockEnvPlayer(last_test, logfile + runtimeId + ".csv", title, seed=seed, commission=commission,
                                    addTA="Y")])
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
        steps = len(np.unique(last_test.date))

        model = PPO2.load("Dan_RL.pkl")
        backtest[loop] = evaluate(model, num_steps=steps)




    return "2"







