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


class LinearTrendPriceGenerator:
    def __init__(self, start_price=10, up=True):
        """
        Initialize the price generator which creates a linear price series starting from `start_price`.
        """
        self.current_price = start_price
        self.up = up

    def generate_price(self, last_price):
        """
        Generate next price in series.
        Parameters:
        last_price (float, optional): This parameter is required to match the interface expected by the simulator,
                                      but it is not used for the linear trend price generation.
        """
        # Since the price is linear, we just increment or decrement based on `up`
        if self.up:
            self.current_price += 1
        else:
            self.current_price -= 1
        
        return self.current_price

    

class CashPriceGenerator:
    """Cash remains constant since returns are sampled from N(0,0)."""
    def generate_price(self, last_price):
        return last_price # No change in value


class NoisyTrendPriceGenerator:
    """Noisy trend price generator using log returns sampled from a normal distribution."""

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
    

class PeriodicTrendPriceGenerator:
    """
    A generator class that produces a periodic sine wave trend
    with adjustable amplitude and frequency, without any noise.

    Attributes:
        start (float): The base value added to the sine function.
        amplitude (float): The amplitude (scaling) of the sine wave.
        frequency (float): The frequency of the sine wave in radians per step.
        t (int): The current time step used in the sine calculation.
    """

    def __init__(self, start=10.0, amplitude=1.0, frequency=1.0):
        """
        Initializes the PeriodicTrendGenerator with a starting value, amplitude, and frequency.

        Args:
            start (float, optional): The base value added to the sine wave. Default is 10.0.
            amplitude (float, optional): The amplitude of the sine wave. Default is 1.0.
            frequency (float, optional): The frequency of the sine wave (radians per step). Default is 1.0.
        """
        self.start = start
        self.amplitude = amplitude
        self.frequency = frequency
        self.t = 0
        
    def generate_price(self, last_price=None):
        """
        Generates the next price using a sine wave and the current time step.

        Parameters:
            last_price (float, optional): This parameter is required to match the interface expected by the simulator,
                                          but it is not used for the linear trend price generation.
        Returns:
            float: The value at time t based on the sine wave and base value.
        """
        value = self.amplitude * np.sin(self.frequency * self.t) + self.start
        self.t += 1
        return value

class NoisyPeriodicTrendPriceGenerator:
    """
    A generator class that produces a periodic sine wave trend
    with adjustable amplitude and frequency, and added Gaussian noise.

    Attributes:
        start (float): The base value added to the sine function.
        amplitude (float): The amplitude (scaling) of the sine wave.
        frequency (float): The frequency of the sine wave in radians per step.
        t (int): The current time step used in the sine calculation.
        mu (float): The mean of the Gaussian noise.
        sigma (float): The standard deviation of the Gaussian noise.
    """

    def __init__(self, start=10.0, amplitude=1.0, frequency=1.0, mu=0.0, sigma=0.2):
        """
        Initializes the PeriodicTrendNoiseGenerator with a starting value,
        amplitude, frequency, and Gaussian noise parameters.

        Args:
            start (float, optional): The base value added to the sine wave. Default is 10.0.
            amplitude (float, optional): The amplitude of the sine wave. Default is 1.0.
            frequency (float, optional): The frequency of the sine wave (radians per step). Default is 1.0.
            mu (float, optional): The mean of the Gaussian noise. Default is 0.0.
            sigma (float, optional): The standard deviation of the Gaussian noise. Default is 0.2.
        """
        self.start = start
        self.amplitude = amplitude
        self.frequency = frequency
        self.t = 0
        self.mu = mu
        self.sigma = sigma

    def generate_price(self, last_price=None):
        """
        Generates the next value in the sequence using a sine function
        scaled by amplitude, modulated by frequency, and added to the base value and noise.
        Parameters:
        last_price (float, optional): This parameter is required to match the interface expected by the simulator,
                                      but it is not used for the linear trend price generation.
        Returns:
            float: The value at time t based on the sine wave, base value, and noise.
        """
        noise = np.random.normal(self.mu, self.sigma)
        value = self.amplitude * np.sin(self.frequency * self.t) + self.start + noise
        self.t += 1
        return value
    
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
    
# Example usage
# generator = NoisyPeriodicTrendPriceGenerator(start=10.0, amplitude=2.0, frequency=0.5, mu=0.0, sigma=0.2)
# for _ in range(5):
#     print(generator.generate())