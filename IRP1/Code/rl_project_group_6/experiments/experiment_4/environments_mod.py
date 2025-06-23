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
    models (e.g., upward trend, downward trend, etc). The generated prices are stored and 
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
        self.days = days 
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

        return df_melted.dropna().reset_index(drop=True)
    
import numpy as np
import pandas as pd
import gym
from gym import spaces
from gym.utils import seeding


class StockPortfolioEnv(gym.Env):
    """
    A portfolio environment for 2 assets: CASH and UPWARD (stock).
    Action = allocation weights [cash, stock]
    State = [cash_price, stock_price]
    Reward = daily portfolio return
    """

    def __init__(self, df, stock_dim, initial_amount, state_space, action_space):
        super(StockPortfolioEnv, self).__init__()

        self.df = df.copy()
        self.stock_dim = stock_dim  # should be 2: [cash, stock]
        self.initial_amount = initial_amount
        self.day = 0
        self.episode = -1

        # Define action and observation space
        self.action_space = spaces.Box(low=0, high=1, shape=(action_space,), dtype=np.float32)

        self.data = self.df[self.df['date'] == self.df['date'].unique()[self.day]].sort_values("tic")
        self.state = self._get_state()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.state.shape, dtype=np.float32)

        # Internal variables
        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.portfolio_return_memory = [0]
        self.date_memory = [self.data.date.values[0]]
        self.episode_log = []

    def _get_state(self):
        state = self.data.sort_values("tic")["close"].values.astype(np.float32)
        return state

    def reset(self):
        self.day = 0
        self.episode += 1
        self.data = self.df[self.df['date'] == self.df['date'].unique()[self.day]].sort_values("tic")
        self.state = self._get_state()

        self.portfolio_value = self.initial_amount
        self.asset_memory = [self.initial_amount]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.portfolio_return_memory = [0]
        self.date_memory = [self.data.date.values[0]]
        self.episode_log = []

        return self.state

    def step(self, actions):
        done = False
        self.day += 1
        if self.day >= len(self.df["date"].unique()):
            done = True
            return self.state, 0.0, done, {}

        # Normalize allocations
        weights = self.normalize_allocation(actions)
        print("weights", actions)
        print("actions", weights)
        # Get today's and previous day's prices
        last_data = self.data
        self.data = self.df[self.df['date'] == self.df['date'].unique()[self.day]].sort_values("tic")
        current_prices = self.data["close"].values
        last_prices = last_data["close"].values

        # Calculate daily return
        portfolio_return = np.dot(((current_prices / last_prices) - 1), weights)
        self.portfolio_value *= (1 + portfolio_return)

        # Log
        self.state = self._get_state()
        self.asset_memory.append(self.portfolio_value)
        self.actions_memory.append(weights.tolist())
        self.portfolio_return_memory.append(portfolio_return)
        self.date_memory.append(self.data.date.values[0])
        cash_weight = weights[0]
        stock_weight = weights[1]
        reward = portfolio_return
        print("State shape:", self.state.shape)
        print("Observation space shape:", self.observation_space.shape)

        self.episode_log.append({
            "episode": self.episode,
            "day": self.day,
            "portfolio_return": portfolio_return,
            "portfolio_value": self.portfolio_value,
            "weights": weights.tolist()
        })

        return self.state, reward, done, {}

    def normalize_allocation(self, actions):
        # Convert arbitrary agent outputs to non-negative values
        actions = np.maximum(actions, 0)  # Ensure no negative allocations
        total = np.sum(actions)

        if total == 0:
            return np.array([0.5, 0.5])
        return actions / total
    
    def save_episode_log(self, filename="episode_log.csv"):
        pd.DataFrame(self.episode_log).to_csv(filename, index=False)
