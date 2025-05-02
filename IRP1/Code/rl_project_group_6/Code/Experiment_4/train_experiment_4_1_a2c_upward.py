
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from finrl import config
from finrl.agents.stablebaselines3.models import DRLAgent
import gym
from gym import spaces
from stable_baselines3.common.vec_env import DummyVecEnv
from generators import LinearTrendPriceGenerator, CashPriceGenerator
from environments import  StockPriceSimulator, StockPortfolioEnvOriginal
import os
from tabulate import tabulate
from plot_functions import preprocess_df, plot_dual_line_per_episode, plot_per_episode_row

initial_prices = {"UPWARD": 100, "CASH": 100}

# Define different price behaviors for each stock
generators = {
"UPWARD": LinearTrendPriceGenerator(start_price=100, up=True),  #
"CASH": CashPriceGenerator()  # Cash remains constant
}

# Create the simulator
simulator = StockPriceSimulator(days=1000, initial_prices=initial_prices, generators=generators)

# Generate stock prices
train = simulator.generate_prices()
train.head()
print(train.loc[train['tic'] == 'UPWARD'])

num_df = train[['close']] #, 'return_t-1', 'return_t-2', 'return_t-3', 'return_t-4', 'return_t-5']]
inf_rows = num_df[np.isinf(num_df).any(axis=1)]
print("Rows containing `inf` values:\n", inf_rows)


# Define the full path for the image
folder_path = "Results/Experiment_4"
image_path = os.path.join(folder_path, "4_1_upward_price_trend.png")

# Create the folder and sub-folder if they do not exist
os.makedirs(folder_path, exist_ok=True)

# Plot stock price
plt.figure(figsize=(12, 6))
for stock in train["tic"].unique():
    if stock != "CASH":
      stock_data = train[train["tic"] == stock]
      plt.plot(stock_data["date"], stock_data["close"], label=stock)

# Formatting the plot
plt.xlabel("Date")
plt.ylabel("Stock Price (Close)")
plt.title("Stock Price Trend")
plt.legend()
plt.grid(True)

# Plot the stock upward trend:
plt.savefig(image_path)


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

A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.005, "learning_rate": 0.0002}
model_a2c = agent.get_model(model_name="a2c",model_kwargs = A2C_PARAMS)

# Define the number of episodes and days and train for episodes*day timesteps


episodes = 500
days = 1000
trained_a2c = agent.train_model(model=model_a2c, 
                                tb_log_name='a2c',
                             total_timesteps=episodes*days)


experiments_settings =  "4_1_a2c_upward"
e_train_gym.save_episode_log(f"Results/Experiment_4/training_logs_{experiments_settings}.csv")

print("Saved successfully")

experiment_number = "4"
# Load data

df_logs = pd.read_csv(f"Results/Experiment_{experiment_number}/training_logs_{experiments_settings}.csv")
df_logs = preprocess_df(df_logs)
# **1️⃣ Portfolio Value Plot**
plot_per_episode_row(df_logs, "new_portfolio_value", "Portfolio Value ($)", "Portfolio Value Across Episodes",
                     f"Results/Experiment_{experiment_number}/{experiments_settings}_portfolio_value.png")

# **2️⃣ Portfolio Return Plot**
plot_per_episode_row(df_logs, "portfolio_return", "Portfolio Return (%)", "Portfolio Return Across Episodes",
                     f"Results/Experiment_{experiment_number}/{experiments_settings}_portfolio_return_row.png")
# **3️⃣ Reward Plot**
plot_per_episode_row(df_logs, "reward", "Reward", "Reward Across Episodes",
                     f"Results/Experiment_{experiment_number}/{experiments_settings}_reward_row.png")

# **1️⃣ Actions Plot**
plot_dual_line_per_episode(df_logs, "actions", "actions", 
                           "Allocation Weights Before Normalization", "Allocation Weights Before Normalization Over Time", 
                           "actions_plot.png", legend1="Action 1", legend2="Action 2")

# **2️⃣ Allocation Weights Plot**
plot_dual_line_per_episode(df_logs, "allocation_weights", "allocation_weights", 
                           "Allocation Weights", "Allocation Over Time", 
                           f"Results/Experiment_{experiment_number}/{experiments_settings}_allocation_plot.png", 
                           legend1="Cash Weight", legend2="Stock Weight")
