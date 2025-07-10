from .generators import BaseGenerator
import numpy as np

# === Wrappers ===
class GaussianNoise(BaseGenerator):
    def __init__(self, generator, mu=0.0, sigma=0.2):
        """
        Wraps any generator and adds Gaussian noise to its output.
        """
        self.generator = generator
        self.mu = mu
        self.sigma = sigma

    def generate_price(self, last_price):
        base_price = self.generator.generate_price(last_price)
        noise = np.random.normal(self.mu, self.sigma)
        return base_price + noise

    def reset(self):
        self.generator.reset()
