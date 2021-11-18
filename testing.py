from env.StockEnvPlayer import StockEnvPlayer

import gym
import numpy as np
import pandas as pd
import sys
import csv
import os

import json
import getopt
import quandl
import talib
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn import preprocessing

from stable_baselines import PPO2  # ,A2C, ACKTR, DQN, DDPG, SAC, PPO1,  TD3, TRPO
from stable_baselines.ddpg import NormalActionNoise
from stable_baselines.common.identity_env import IdentityEnv, IdentityEnvBox
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import MlpPolicy


# def evaluate(model, num_steps=1000):
#     episode_rewards = [0.0]
#     obs = env.reset()
#     env.render()
#
#     for i in range(num_steps):
#         action, _states = model.predict(obs)
#         obs, rewards, done, info = env.step(action)
#         env.render()
#
#         # Stats
#         episode_rewards[-1] += rewards
#         if done:
#             obs = env.reset()
#             episode_rewards.append(0.0)
#
#     return np.sum(episode_rewards)

algo = PPO2

# global env


seed = 42
lr = 1e-2
cliprange = 0.3
g = 0.99

set_global_seeds(seed)
np.random.seed(seed)


env = DummyVecEnv([lambda: StockEnvPlayer(seed=seed, commission=0, addTA="Y")])
env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
# steps = len(np.unique(test.date))
# backtest[loop] = evaluate(model, num_steps=steps)


model = algo(
    MlpPolicy,
    env,
    gamma=g,
    n_steps=128,
    ent_coef=0.01,
    learning_rate=lr,
    vf_coef=0.5,
    max_grad_norm=0.5,
    lam=0.95,
    nminibatches=4,
    noptepochs=4,
    cliprange=cliprange,
    cliprange_vf=None,  # tensorboard_log="./tensorlog",
    _init_setup_model=True,
    policy_kwargs=None,
    full_tensorboard_log=False,
)

model.load("Dan_RL")
