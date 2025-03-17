import numpy as np
import pandas as pd
from tabulate import tabulate
import gym
from gym import spaces
from gym.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv

class PortfolioEnvironment:
    def __init__(self, generator, agent, initial_cash=500, initial_stock_value=500, start_price=10):
        """
        Initialize the trading environment.

        Parameters:
        - generator: An instance that generates market prices.
        - agent: An RL or rule-based agent.
        - n: Number of time steps per episode.
        - initial_cash: Starting cash balance.
        - initial_stock_value: Starting stock value.
        - start_price: Initial asset price.
        """
        self.generator = generator  # Price generator instance
        self.agent = agent  # Agent instance

        # Initialize portfolio
        self.initial_cash = initial_cash
        self.initial_stock_value = initial_stock_value
        self.start_price = start_price

        self.reset()

    def reset(self):
        """ Reset the environment for a new episode. """
        self.current_step = 0
        self.current_price = self.start_price
        self.stock_value = self.initial_stock_value
        self.cash = self.initial_cash
        self.pv = self.stock_value + self.cash
        self.allocation = self.stock_value / self.pv

        self.prices = [self.current_price]
        self.pvs = [self.pv]
        self.allocations = [self.allocation]
        self.actions = []
        self.rewards = []

        return self._get_state()

    def step(self):
        """ Run one iteration of the environment (one time step). """

        self.current_step += 1

        # Generate new price from the generator
        new_price = self.generator.generate_price()
        return_ratio = new_price / self.current_price
        self.current_price = new_price
        self.prices.append(self.current_price)

        # Compute new portfolio value
        self.stock_value *= return_ratio
        self.pv = self.stock_value + self.cash
        self.allocation = self.stock_value / self.pv

        # Get state for the agent (e.g., price up/down)
        state = 1 if return_ratio > 1 else 0

        # Get action from the agent
        action = self.agent.act(state)
        self.actions.append(action)

        # Convert action into allocation change, 0 is hold, 1 is buy, 2 is sell,
        if action == 1 and self.allocation < 0.9:
            target_allocation = self.allocation + 0.1  # Buy more stock
        elif action == 2 and self.allocation > 0.1:
            target_allocation = self.allocation - 0.1  # Sell stock
        else:
            target_allocation = self.allocation  # Hold

        # Adjust portfolio
        self.stock_value = self.pv * target_allocation
        self.cash = self.pv - self.stock_value
        self.allocation = self.stock_value / self.pv
        self.allocations.append(self.allocation)

        # Calculate reward
        reward = self.pv - self.pvs[-1]
        self.pvs.append(self.pv)
        self.rewards.append(reward)
        

        # Let the agent learn
        next_state = 1 if np.random.rand() > 0.5 else 0  # Random future state
        self.agent.learn(state, action, reward, next_state)

        return state, reward

    def _get_state(self):
        """ Returns the initial state (price movement direction). """
        return 1 if len(self.prices) > 1 and self.prices[-1] > self.prices[-2] else 0



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

        for t in range(1, 6):  # Compute past returns up to t-5
            df_melted[f"return_t-{t}"] = df_melted.groupby("tic")["close"].pct_change(t)

        return df_melted.dropna().reset_index(drop=True)
    

class StockPortfolioEnv(gym.Env):
    """A stock trading environment for OpenAI Gym.

    This environment simulates stock trading, allowing an agent to allocate 
    capital among different assets and optimize portfolio returns.

    Attributes:
    -----------
        df (DataFrame): Data containing stock prices and returns.
        stock_dim (int): Number of unique stocks.
        initial_amount (float): Initial capital.
        transaction_cost_pct (float): Cost of trading as a percentage.
        reward_function (str): Defines reward calculation ("dense" or "sparse").
        reward_scaling (float): Scaling factor for rewards.
        state_space (int): Number of features in the observation space.
        action_space (int): Number of available actions (stock allocations).
        past_returns (list): List of past return columns.
        timestep_log (bool): If True, logs actions, allocations, and rewards.
        day (int): Tracks the current day of the episode.
        episode (int): Tracks the episode number.
        portfolio_value (float): Tracks the current value of the portfolio.
        terminal (bool): Indicates if the episode has ended.
        asset_memory (list): Stores portfolio value over time.
        portfolio_return_memory (list): Stores portfolio returns.
        actions_memory (list): Stores past action allocations.
        date_memory (list): Stores dates corresponding to each step.
        episode_log (list): Stores detailed episode history.

    Methods:
    --------
        _get_state(): Retrieves the current state of the environment.
        softmax_normalization(actions): Normalizes action values.
        step(actions): Executes a trading action and updates state.
        reset(): Resets the environment for a new episode.
        save_episode_log(filename): Saves episode logs to a CSV file.
        get_sb_env(): Returns a stable-baselines3 compatible environment.
    """

    metadata = {'render.modes': ['human']}
    
    def __init__(self,
                 df,
                 stock_dim,
                 initial_amount,
                 transaction_cost_pct,
                 reward_function="time_step_return",  
                 reward_scaling=1.0,
                 state_space=10,
                 action_space=10,
                 past_returns=None,
                 timestep_log=False,
                 day=-1):
        """
        Initializes the stock trading environment.

        Parameters:
        -----------
            df (DataFrame): Stock price and return data.
            stock_dim (int): Number of stocks.
            initial_amount (float): Starting capital.
            transaction_cost_pct (float): Trading cost percentage.
            reward_function (str): Type of reward function.
            reward_scaling (float): Scaling factor for rewards.
            state_space (int): Number of state features.
            action_space (int): Number of actions.
            past_returns (list): Past return columns for observations.
            timestep_log (bool): If True, logs each timestep.
            day (int): Starting day index (-1 to initialize properly).
        """
        self.episode = -1  # Episode counter
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.reward_function = reward_function 
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.past_returns = past_returns
        self.timestep_log = timestep_log

        # Define action and observation spaces
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                            shape=(self.state_space + len(self.past_returns), self.state_space))
      
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
        state_columns = ["close"] + self.past_returns  
        state_array = data_sorted[state_columns].to_numpy()  
        return state_array.flatten()

    def softmax_normalization(self, actions):
        """Applies softmax transformation to normalize actions."""
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        return numerator / denominator
    
    def normalize_actions(self, actions):
        actions = np.array(actions, dtype=np.float64)  
        
        if np.all(actions == actions[0]):  # If all values are the same (e.g., [0,0] or [1,1])
            norm_actions = np.ones_like(actions) / len(actions)  # Equal allocation
        else:
            min_adjusted = actions - actions.min()  # Shift to make the smallest value zero
            norm_actions = min_adjusted / min_adjusted.sum()  # Normalize to sum to 1

        return norm_actions

    def step(self, actions):
        """Executes a step in the environment.

        Parameters:
        -----------
            actions (list or array): Portfolio allocations.

        Returns:
        --------
            state (array): The updated environment state.
            reward (float): The reward for the step.
            terminal (bool): Whether the episode has ended.
            {} (dict): Empty dictionary for compatibility.
        """
        # Initial step: Allocate 50-50 if starting fresh
        if self.day == -1:
            actions = [0.5, 0.5]

        # Normalize actions 
        weights = self.normalize_actions(actions)

        # Compute transaction cost
        if len(self.actions_memory) > 0:
            transaction_cost = self.transaction_cost_pct * np.sum(np.abs(weights - self.actions_memory[-1])) * self.portfolio_value
        else:
            transaction_cost = 0  

        # Store action
        self.actions_memory.append(weights)

        # Update time step
        self.day += 1

        # Check if it is the end of episode
        self.terminal = self.day >= len(self.df["date"].unique()) - 1

        # Filter the dataframe for the correct date
        self.data = self.df[self.df["date"] == self.df["date"].unique()[self.day]]

        # Get state
        self.state = self._get_state()

        # Calculate portfolio return from stock return
        stock_returns = self.data["return_t-1"].values
        portfolio_return = sum(stock_returns * weights)

        # Calculate new portfolio value
        new_portfolio_value = self.portfolio_value * (1 + portfolio_return) - transaction_cost
        
        # Calculate reward
        if self.reward_function == "sparse":
            self.reward = new_portfolio_value - self.asset_memory[0] if self.terminal else -transaction_cost
        elif self.reward_function == "dense":
            self.reward = new_portfolio_value - self.portfolio_value - transaction_cost
        else:
            raise ValueError("Invalid reward function. Choose 'final_portfolio_value' or 'time_step_return'.")

        if self.timestep_log == True:
            # Convert NumPy arrays to lists and format values
            table= [
                ["Episode", self.episode],
                ["Day", self.day],
                ["Actions (weights before normalization)", 
                ", ".join([f"{x:.2f}" for x in actions.tolist()]) if isinstance(actions, np.ndarray) else actions],
                ["Allocation weights", 
                ", ".join([f"{x:.2f}" for x in weights.tolist()]) if isinstance(weights, np.ndarray) else weights],
                ["Transaction Cost", f"{transaction_cost:.2f}"],
                ["Stock Returns", 
                ", ".join([f"{x:.2%}" for x in stock_returns.tolist()]) if isinstance(stock_returns, np.ndarray) else stock_returns],
                ["Portfolio Return", f"{portfolio_return:.2%}"],
                ["Reward", f"{self.reward:.2f}"],
                ["Old Portfolio Value", f"{self.portfolio_value:,.0f}"],  
                ["New Portfolio Value", f"{(self.portfolio_value * (1 + portfolio_return) - transaction_cost):,.0f}"]
            ]

            # Print a separator line before each table for better readability
            print("\n" + "=" * 53)  # Separator line
            print(tabulate(table, tablefmt="grid"))  # Print table without headers
            print("=" * 53 + "\n")  # Separator line after table

        # Store historical data
        self.portfolio_return_memory.append(portfolio_return)
        self.date_memory.append(self.data.date.unique()[0])
        self.asset_memory.append(new_portfolio_value)
        
        self.episode_log.append({
            "episode": self.episode,
            "day": self.day,
            "actions": ", ".join([f"{x:.2f}" for x in actions.tolist()]) if isinstance(actions, np.ndarray) else actions,
            "allocation_weights": ", ".join([f"{x:.2f}" for x in weights.tolist()]) if isinstance(weights, np.ndarray) else weights,
            "transaction_cost": f"{transaction_cost:.2f}",
            "stock_returns": ", ".join([f"{x:.2%}" for x in stock_returns.tolist()]) if isinstance(stock_returns, np.ndarray) else stock_returns,
            "portfolio_return": f"{portfolio_return:.2%}",
            "reward": f"{self.reward:.2f}",
            "old_portfolio_value": f"{self.portfolio_value:,.0f}",
            "new_portfolio_value": f"{(self.portfolio_value * (1 + portfolio_return) - transaction_cost):,.0f}"
        })
        self.portfolio_value =new_portfolio_value

        return self.state, self.reward, self.terminal, {}
       

    def reset(self):
        """Resets the environment for a new episode."""
        self.episode += 1  
        print(f"\033[1;34mResetting environment for Episode {self.episode}.\033[0m")
        self.asset_memory = [self.initial_amount]
        self.day = -1
        self.data = self.df[self.df["date"] == self.df["date"].unique()[self.day]]
        self.state = self._get_state()
        self.portfolio_value = self.initial_amount
        self.terminal = False
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / self.stock_dim] * self.stock_dim]
        self.date_memory = [self.data.date.unique()[0]]
    
        return self.state
    
    def save_episode_log(self, filename="episode_log.csv"):
        """Save episode logs to a CSV file."""
        df_log = pd.DataFrame(self.episode_log)
        df_log.to_csv(filename, index=False)
        print(f"Saved episode log to {filename}")


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        print("EPISODE: ", self.episode)
        return e, obs