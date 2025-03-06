import numpy as np

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