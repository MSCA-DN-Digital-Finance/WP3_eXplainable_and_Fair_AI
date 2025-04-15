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
        return last_price # No change in value


class UpwardTrendPriceGenerator:
    """Upward trend price generator using log returns sampled from a normal distribution."""

    def __init__(self, mean=0.001, variance=0.001):
        """
        Initialize the generator with a mean and variance for log returns.

        Parameters:
        - mean: Mean of the log return distribution (default 0.001).
        - variance: Variance of the log return distribution (default 0.001).
        """
        self.mean = mean
        self.variance = variance

    def generate_price(self, last_price):
        """
        Generate the next price using log returns from N(mean, variance).

        Parameters:
        - last_price: Previous stock price.

        Returns:
        - new_price: Updated stock price after applying the log return.
        """
        log_return = np.random.normal(self.mean, self.variance)  # Sample log return
        new_price = last_price * np.exp(log_return)  # Apply log return
        return new_price  # Prevent negative prices