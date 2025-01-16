import numpy as np
from tqdm import tqdm

class Experiment:
    """
    A class to run multiple iterations of a game simulation.

    Attributes:
        n_runs (int): Number of simulation runs.
        game_class (class): Class implementing the game logic.
        generator_class (class): Class implementing the generator logic.
        observer_class (class): Class implementing the observer logic.
        T (int): Number of time steps in the game.
        initial_values (dict): Initial values or parameters for the game.
        initial_portfolio (float): Starting portfolio value (default is 100).
    """

    def __init__(self, n_runs, game_class, generator_class, observer_class, T, initial_values, initial_portfolio=100):
        """
        Initializes the Experiment with the specified parameters.

        Args:
            n_runs (int): Number of simulation runs.
            game_class (class): Class implementing the game logic.
            generator_class (class): Class implementing the generator logic.
            observer_class (class): Class implementing the observer logic.
            T (int): Number of time steps in the game.
            initial_values (dict): Initial values or parameters for the game.
            initial_portfolio (float, optional): Starting portfolio value. Default is 100.
        """
        self.n_runs = n_runs
        self.game_class = game_class
        self.generator_class = generator_class
        self.observer_class = observer_class
        self.T = T
        self.initial_values = initial_values
        self.initial_portfolio = initial_portfolio

    def execute(self):
        """
        Executes the experiment by running the game multiple times.

        Returns:
            numpy.ndarray: An array containing the results of each run.
        """
        results = np.array([])  # Initialize an empty NumPy array to store results.

        for _ in tqdm(range(self.n_runs), desc="Running simulations"):
            # Instantiate the observer, generator, and game classes for each run.
            observer = self.observer_class()
            generator = self.generator_class()
            game = self.game_class(generator, observer, self.T, self.initial_values, self.initial_portfolio)

            # Execute the game and collect the result.
            result = game.play_game()
            
            # Append the result to the results array.
            results = np.append(results, result)

        return results

# Example usage
# Uncomment and replace with actual classes to test the functionality.
# experiment = Experiment(
#     n_runs=100,
#     game_class=GameClass,
#     generator_class=GeneratorClass,
#     observer_class=ObserverClass,
#     T=100,
#     initial_values=1,
#     initial_portfolio=100
# )
# results = experiment.execute()
