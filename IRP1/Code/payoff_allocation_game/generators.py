import numpy as np



class LinearGenerator:
    """
    A simple generator class that produces a sequence of linearly increasing values.

    Attributes:
        value (int): The current value of the generator, initialized to the start value.
    """

    def __init__(self, start=2):
        """
        Initializes the LinearGenerator with a starting value.

        Args:
            start (int, optional): The initial value of the generator. Default is 2.
        """
        self.value = start

    def generate(self):
        """
        Generates the next value in the sequence and increments the current value.

        Returns:
            int: The current value before incrementing.
        """
        value = self.value
        self.value += 1
        return value

# Example usage
# Uncomment the following lines to test the LinearGenerator functionality.
# generator = LinearGenerator(start=10)
# for _ in range(5):
#     print(generator.generate())







class GaussianGenerator:
    """
    A generator class that produces values sampled from a Gaussian (normal) distribution.

    Attributes:
        mu (float): Mean of the Gaussian distribution.
        sigma (float): Standard deviation of the Gaussian distribution.
    """

    def __init__(self, mu, sigma):
        """
        Initializes the GaussianGenerator with specified mean and standard deviation.

        Args:
            mu (float): Mean of the Gaussian distribution.
            sigma (float): Standard deviation of the Gaussian distribution.
        """
        self.mu = mu
        self.sigma = sigma

    def generate(self):
        """
        Generates a random value sampled from the Gaussian distribution.

        Returns:
            float: A value sampled from the Gaussian distribution with the specified mean and standard deviation.
        """
        return np.random.normal(self.mu, self.sigma)

# Example usage
# Uncomment the following lines to test the GaussianGenerator functionality.
# generator = GaussianGenerator(mu=0, sigma=1)
# for _ in range(5):
#     print(generator.generate())
