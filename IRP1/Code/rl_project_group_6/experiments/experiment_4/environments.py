import numpy as np
import pandas as pd
import gym
from gym import spaces
from gym.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv


class StockPriceSimulator:
    """
    Simulates stock prices over a specified time period using different price generators.

    This class allows users to simulate stock price movements using various price generation 
    models (e.g., log-normal returns, random walks, etc). The generated prices are stored and 
    processed into a structured pandas DataFrame for further analysis.

    Attributes:
    -----------
    days : int
        The total number of days to simulate.
    initial_prices : dict
        A dictionary containing the initial prices of each stock, where keys are stock tickers 
        and values are their respective starting prices.
    generators : dict
        A dictionary mapping stock tickers to their respective price generators. Each generator 
        is expected to have a `generate_price(last_price)` method to simulate the next price.
    stock_prices : dict
        A dictionary storing the simulated price series for each stock.

    Methods:
    --------
    generate_prices():
        Runs the simulation for the given number of days, updating prices using the 
        specified generators.
        
    to_dataframe():
        Converts the stored price data into a structured pandas DataFrame, reshaped for 
        easier analysis. Computes past returns up to 5 time steps for each stock.
    
    Returns:
    --------
    - A pandas DataFrame containing:
        - `date`: The date of each price observation.
        - `tic`: The stock ticker symbol.
        - `close`: The simulated closing price.
        - `return_t-1` to `return_t-5`: Percentage price changes over previous time steps.
    """
    def __init__(self, days, initial_prices, generators, seed=42):
        np.random.seed(seed)
        self.days = days + 5
        self.initial_prices = initial_prices
        self.generators = generators
        self.stock_prices = {stock: [initial_prices[stock]] for stock in initial_prices.keys()}
        
    def generate_prices(self):
        """Simulate stock prices over the given time period."""
        for _ in range(1, self.days):
            for stock, generator in self.generators.items():
                new_price = generator.generate_price(self.stock_prices[stock][-1])
                self.stock_prices[stock].append(new_price)

        return self.to_dataframe()

    def to_dataframe(self):
        """Convert stock price data into a pandas DataFrame."""
        df = pd.DataFrame({"date": pd.date_range(start="2023-01-01", periods=self.days)})
        for stock in self.initial_prices.keys():
            df[stock] = self.stock_prices[stock]

        df_melted = df.melt(id_vars=["date"], var_name="tic", value_name="close")
        df_melted.sort_values(by=["tic", "date"], inplace=True)

        grouped = df_melted.groupby("tic")

        # 1. Lagged returns (already present)
        for t in range(1, 6):
            df_melted[f"return_t-{t}"] = grouped["close"].pct_change(t)
        # Momentum at 1, 3, 5, 10 days
        df_melted["momentum_1"] = grouped["close"].diff(1)
        df_melted["momentum_3"] = grouped["close"].diff(3)
        df_melted["momentum_5"] = grouped["close"].diff(5)
        df_melted["momentum_10"] = grouped["close"].diff(10)

        # Rate of change
        df_melted["roc_3"] = grouped["close"].pct_change(3)
        df_melted["roc_10"] = grouped["close"].pct_change(10)

        return df_melted.dropna().reset_index(drop=True)
    
    
class StockPortfolioEnv(gym.Env):
    """A single stock trading environment for OpenAI gym

    Attributes
    ----------
        df: DataFrame
            input data
        stock_dim : int
            number of unique stocks
        initial_amount : int
            start money
        state_space: int
            the dimension of input features
        action_space: int
            equals stock dimension
        day: int
            an increment number to control date

    Methods
    -------
    step()
        at each step the agent will return actions, then 
        we will calculate the reward, and return the next observation.
    reset()
        reset the environment
    """

    def __init__(self, 
                df,
                stock_dim,
                initial_amount,
                state_space,
                action_space,
                day = 0):

        self.episode = -1 
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.initial_amount = initial_amount
        self.state_space = state_space
        self.action_space = action_space
        self.reward = 0
       # Define action and observation spaces
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_space,))
        #added from here
        self.data = self.df[self.df['date'] == self.df['date'].unique()[self.day]].sort_values("tic")
        self.state = self._get_state()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.state.shape, dtype=np.float32)
        # to here
        self.reward_mean = 0
        self.reward_var = 1
        self.alpha = 0.01  # smoothing

        # Initialize logs
        self.episode_log = []  

        # Load first day's data
        self.data = self.df[self.df["date"] == self.df["date"].unique()[self.day]]
        self.state = self._get_state()  
        self.terminal = False
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]

    def _get_state(self):
        """Retrieves the current state based on stock prices and past returns."""
        data_sorted = self.data.sort_values("tic")  

        state_columns = ['close', 'return_t-1', 'return_t-2', 'return_t-3', 'return_t-4', 'return_t-5', 'momentum_1', 'momentum_3', 'momentum_5', 'momentum_10', 'roc_3', 'roc_10'] 
        state_array = data_sorted[state_columns].to_numpy()  

        return state_array.flatten()
    def update_reward_stats(self, reward):
        self.reward_mean = (1 - self.alpha) * self.reward_mean + self.alpha * reward
        self.reward_var = (1 - self.alpha) * self.reward_var + self.alpha * (reward - self.reward_mean) ** 2

    def normalize_reward(self, reward):
        return (reward - self.reward_mean) / (self.reward_var ** 0.5 + 1e-8)

    def step(self, actions):
     
        self.terminal = self.day >= len(self.df["date"].unique()) - 1

        if self.terminal:

            return self.state, self.reward, self.terminal,{}

        else:
            if self.day == 0:
                actions = [0.5, 0.5]
              
            weights = self.normalize_allocation(actions) 
            print("raw actions ", actions)
            print("weights: ", weights)
            self.actions_memory.append(weights)
            last_day_memory = self.data

            #load next state
            self.day += 1
            self.data = self.df[self.df["date"] == self.df["date"].unique()[self.day]]
            self.state =  self._get_state()
            portfolio_return = sum(((self.data.close.values / last_day_memory.close.values)-1)*weights)
            # update portfolio value
            new_portfolio_value = self.portfolio_value*(1+portfolio_return)
            self.portfolio_value = new_portfolio_value

            # save into memory
            self.portfolio_return_memory.append(portfolio_return)
            self.date_memory.append(self.data.date.unique()[0])            
            self.asset_memory.append(new_portfolio_value)
            
        
            price_today = self.data.close.values[1]  # stock
            price_yesterday = last_day_memory.close.values[1]

            # Upward momentum → buy stock; downward → move to cash
            momentum = price_today - price_yesterday
            self.reward = weights[1] * momentum


            print("reward", self.reward)

        self.episode_log.append({
        "episode": self.episode,
        "day": self.day,
        "actions": ", ".join([f"{x:.2f}" for x in actions]) if isinstance(actions, (np.ndarray, list)) else actions,
        "allocation_weights": ", ".join([f"{x:.2f}" for x in weights.tolist()]) if isinstance(weights, np.ndarray) else weights,
        "portfolio_return": f"{portfolio_return:.2%}",
        "reward": f"{self.reward:.2f}",
        "new_portfolio_value": f"{new_portfolio_value:,.0f}"
        })
        return self.state, self.reward, self.terminal, {}

    def reset(self):
        self.episode += 1  
        print("EPISODE: ", self.episode)
        self.asset_memory = [self.initial_amount]

        self.day = 0
        self.data = self.df[self.df["date"] == self.df["date"].unique()[self.day]]
        # load states
        self.state =  self._get_state()  
        self.portfolio_value = self.initial_amount

        self.terminal = False 
        self.portfolio_return_memory = [0]
        self.actions_memory=[[1/self.stock_dim]*self.stock_dim]
        self.reward = 0
        self.date_memory=[self.data.date.unique()[0]] 
        return self.state
        
    def normalize_allocation(self, actions):
        # Convert arbitrary agent outputs to non-negative values
        actions = np.maximum(actions, 0)  # Ensure no negative allocations
        total = np.sum(actions)

        if total == 0:
            return np.array([0.5, 0.5])
        return actions / total

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
    
    def save_episode_log(self, filename="episode_log.csv"):
        """Save episode logs to a CSV file."""
        df_log = pd.DataFrame(self.episode_log)
        df_log.to_csv(filename, index=False)
        print(f"Saved episode log to {filename}")