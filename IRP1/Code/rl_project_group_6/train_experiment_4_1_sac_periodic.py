
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from finrl import config
from finrl.agents.stablebaselines3.models import DRLAgent
import gym
from gym import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
from generators import PeriodicTrendPriceGenerator, CashPriceGenerator
from environments import  StockPriceSimulator, StockPortfolioEnvOriginal
import os
from tabulate import tabulate

initial_prices = {"UPWARD": 100, "CASH": 100}

# Define different price behaviors for each stock
generators = {
"UPWARD": PeriodicTrendPriceGenerator(10,1,1), #
"CASH": CashPriceGenerator()  # Cash remains constant
}


# Create the simulator
simulator = StockPriceSimulator(days=1000, initial_prices=initial_prices, generators=generators)

# Generate stock prices
train = simulator.generate_prices()
train.head()
train.loc[train['tic'] == 'UPWARD']

num_df = train[['close']] #, 'return_t-1', 'return_t-2', 'return_t-3', 'return_t-4', 'return_t-5']]
inf_rows = num_df[np.isinf(num_df).any(axis=1)]
print("Rows containing `inf` values:\n", inf_rows)


stock_dimension = len(train["tic"].unique())  # Count unique stocks
#state_space = len(["close", "return_t-1", "return_t-2", "return_t-3", "return_t-4", "return_t-5"]) * stock_dimension
state_space = len(["close"]) * stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")
env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000, 
    "transaction_cost_pct": 0.001,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "action_space": stock_dimension,
    "reward_scaling":  1e-4,
    #"reward_function": "sparse",
}

# Create the environment
e_train_gym = StockPortfolioEnvOriginal(df=train, **env_kwargs)

env_train, _ = e_train_gym.get_sb_env()


# initialize the agent
agent = DRLAgent(env = env_train)
SAC_PARAMS = {
    "batch_size": 128,
    "buffer_size": 100000,
    "learning_rate": 0.0003,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}

model_sac = agent.get_model("sac",model_kwargs = SAC_PARAMS)

# Define the number of episodes and days and train for episodes*day timesteps


episodes = 500
days = 1000
trained_sac = agent.train_model(model=model_sac, 
                             tb_log_name='sac',
                             total_timesteps=episodes*days)
algorithm = ""
e_train_gym.save_episode_log("Results/Experiment_4/training_logs_4_1_sac_periodic.csv")

print("Saved successfully")
