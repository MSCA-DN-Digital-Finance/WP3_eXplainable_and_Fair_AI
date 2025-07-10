import numpy as np
from abc import ABC, abstractmethod

# === Base Class ===
class BaseGenerator(ABC):
    @abstractmethod
    def generate_price(self, last_price):
        """Generate the next price in the series."""
        pass

    def reset(self):
        """Optional reset method."""
        pass

# === Core Generators ===
class LinearTrendGenerator(BaseGenerator):
    def __init__(self, start_price=10.0, up=True):
        self.initial_price = start_price
        self.current_price = start_price
        self.up = up

    def generate_price(self, last_price=None):
        self.current_price += 1 if self.up else -1
        return self.current_price

    def reset(self):
        self.current_price = self.initial_price

class ConstantGenerator(BaseGenerator):
    def generate_price(self, last_price):
        return last_price  # Always constant

class PeriodicTrendGenerator(BaseGenerator):
    def __init__(self, start=10.0, amplitude=1.0, frequency=1.0):
        self.start = start
        self.amplitude = amplitude
        self.frequency = frequency
        self.t = 0

    def generate_price(self, last_price=None):
        value = self.amplitude * np.sin(self.frequency * self.t) + self.start
        self.t += 1
        return value

    def reset(self):
        self.t = 0
