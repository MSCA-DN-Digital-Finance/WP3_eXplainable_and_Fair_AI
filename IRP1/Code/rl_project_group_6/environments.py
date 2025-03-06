import numpy as np

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
