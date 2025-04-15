import numpy as np



class LinearUpTrendGenerator:
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


class LinearUpTrendNoiseGenerator:
    """
    A generator class that produces a sequence of linearly increasing values
    with added Gaussian noise.

    Attributes:
        value (float): The current value of the generator, initialized to the start value.
        mu (float): The mean of the Gaussian noise.
        sigma (float): The standard deviation of the Gaussian noise.
    """

    def __init__(self, start=10.0, mu=0.0, sigma=0.2):
        """
        Initializes the LinearUpTrendNoiseGenerator with a starting value
        and Gaussian noise parameters.

        Args:
            start (float, optional): The initial value of the generator. Default is 10.0.
            mu (float, optional): The mean of the Gaussian noise. Default is 0.0.
            sigma (float, optional): The standard deviation of the Gaussian noise. Default is 0.2.
        """
        self.value = start
        self.mu = mu
        self.sigma = sigma

    def generate(self):
        """
        Generates the next value in the sequence by increasing the current value
        by 1 plus added Gaussian noise.

        Returns:
            float: The updated value after applying the linear increment and noise.
        """
        noise = np.random.normal(self.mu, self.sigma)
        self.value = self.value + 1 + noise
        return self.value

# Example usage
# Uncomment the following lines to test the LinearUpTrendNoiseGenerator functionality.
# generator = LinearUpTrendNoiseGenerator(start=10.0, mu=0.0, sigma=0.2)
# for _ in range(5):
#     print(generator.generate())


class LinearDownTrendGenerator:
    """
    A simple generator class that produces a sequence of linearly decreasing values.

    Attributes:
        value (int): The current value of the generator, initialized to the start value.
    """

    def __init__(self, start=10):
        """
        Initializes the LinearDownTrendGenerator with a starting value.

        Args:
            start (int, optional): The initial value of the generator. Default is 10.
        """
        self.value = start

    def generate(self):
        """
        Generates the next value in the sequence and decrements the current value.

        Returns:
            int: The current value before decrementing.
        """
        value = self.value
        self.value -= 1
        return value

# Example usage
# Uncomment the following lines to test the LinearDownTrendGenerator functionality.
# generator = LinearDownTrendGenerator(start=10)
# for _ in range(5):
#     print(generator.generate())



class LinearDownTrendNoiseGenerator:
    """
    A generator class that produces a sequence of linearly decreasing values
    with added Gaussian noise.

    Attributes:
        value (float): The current value of the generator, initialized to the start value.
        mu (float): The mean of the Gaussian noise.
        sigma (float): The standard deviation of the Gaussian noise.
    """

    def __init__(self, start=10.0, mu=0.0, sigma=0.2):
        """
        Initializes the LinearDownTrendNoiseGenerator with a starting value
        and Gaussian noise parameters.

        Args:
            start (float, optional): The initial value of the generator. Default is 10.0.
            mu (float, optional): The mean of the Gaussian noise. Default is 0.0.
            sigma (float, optional): The standard deviation of the Gaussian noise. Default is 0.2.
        """
        self.value = start
        self.mu = mu
        self.sigma = sigma

    def generate(self):
        """
        Generates the next value in the sequence by decreasing the current value
        by 1 plus added Gaussian noise.

        Returns:
            float: The updated value after applying the linear decrement and noise.
        """
        noise = np.random.normal(self.mu, self.sigma)
        self.value = self.value - 1 + noise
        return self.value

# Example usage
# Uncomment the following lines to test the LinearDownTrendNoiseGenerator functionality.
# generator = LinearDownTrendNoiseGenerator(start=10.0, mu=0.0, sigma=0.2)
# for _ in range(5):
#     print(generator.generate())



class PeriodicTrendGenerator:
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

    def generate(self):
        """
        Generates the next value in the sequence using a sine function
        scaled by amplitude, modulated by frequency, and added to the base value.

        Returns:
            float: The value at time t based on the sine wave and base value.
        """
        value = self.amplitude * np.sin(self.frequency * self.t) + self.start
        self.t += 1
        return value

# Example usage
# generator = PeriodicTrendGenerator(start=10.0, amplitude=2.0, frequency=0.5)
# for _ in range(5):
#     print(generator.generate())



class PeriodicTrendNoiseGenerator:
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

    def generate(self):
        """
        Generates the next value in the sequence using a sine function
        scaled by amplitude, modulated by frequency, and added to the base value and noise.

        Returns:
            float: The value at time t based on the sine wave, base value, and noise.
        """
        noise = np.random.normal(self.mu, self.sigma)
        value = self.amplitude * np.sin(self.frequency * self.t) + self.start + noise
        self.t += 1
        return value

# Example usage
# generator = PeriodicTrendNoiseGenerator(start=10.0, amplitude=2.0, frequency=0.5, mu=0.0, sigma=0.2)
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


