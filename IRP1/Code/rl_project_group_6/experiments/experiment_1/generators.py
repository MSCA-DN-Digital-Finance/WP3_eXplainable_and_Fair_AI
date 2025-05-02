import numpy as np

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
    

class CashPriceGenerator:
    """Cash remains constant since returns are sampled from N(0,0)."""
    def generate_price(self, last_price):
        return last_price # No change in value

