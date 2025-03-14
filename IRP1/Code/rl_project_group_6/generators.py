import numpy as np
import pandas as pd

class NormalPriceGenerator:
    def __init__(self, mu, sigma):
        """
        Initialize the price generator with given mean (mu) and standard deviation (sigma).
        
        Parameters:
        - mu: Mean of the normal distribution.
        - sigma: Standard deviation of the normal distribution.
        """
        self.mu = mu
        self.sigma = sigma

    def generate_price(self):
        """
        Generate a random price from a normal distribution.
        
        Returns:
        - A single random price sampled from N(mu, sigma).
        """
        return np.random.normal(self.mu, self.sigma)


class LinearPriceGenerator:
    def __init__(self, start_price=10):
        """
        Initialize the price generator which creates a linear price series starting from 1.
        """
        self.current_price = start_price


    def generate_price(self):
        """
        Generate next price in series.
        """
        self.current_price += 1
        return self.current_price
    

class CashPriceGenerator:
    """Cash remains constant since returns are sampled from N(0,0)."""
    def generate_price(self, last_price):
        return last_price  # No change in value


class UpwardTrendPriceGenerator:
    """Upward trend price generator with returns sampled from N(0.1, 2)."""
    def __init__(self, mu=0.1, sigma=2):
        self.mu = mu
        self.sigma = sigma

    def generate_price(self, last_price):
        """Generate the next price using additive returns from N(0.1, 2)."""
        r = np.random.normal(self.mu, self.sigma)  # Sample return
        new_price = last_price + r  # Additive price update
        return max(0.001, new_price)  # Prevent negative prices


class StockPriceSimulator:
    """Simulates stock prices using different price generators."""
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

        for t in range(1, 6):  # Compute past returns up to t-5
            df_melted[f"return_t-{t}"] = df_melted.groupby("tic")["close"].pct_change(t)

        return df_melted.dropna().reset_index(drop=True)