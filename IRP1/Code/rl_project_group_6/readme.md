# Reinforcement Learning Project: Portfolio Management

This repository implements and experiments with reinforcement learning (RL) strategies for portfolio management. It is structured to facilitate modular development and testing of various RL agents and market environments.

## Repository Structure

- **`agents.py`**: Implements different RL agents, such as Q-Learning and baseline agents, which interact with the environment to learn optimal portfolio management strategies.

- **`environments.py`**: Defines the market environments in which the agents operate, including the dynamics of asset prices and portfolio valuations.

- **`generators.py`**: Implements various data generators that simulate market conditions, providing the environments with synthetic data for training and testing the agents.

-  **`plot_functions.py`**: Defines various functions that are used to plot the results of the experiments

- **Notebooks**:  These Jupyter notebooks are used to run experiments, analyze results, and compare the performance of different agents and strategies under various market scenarios.
  **date_train_experiments.ipynb**: This notebook is used to train all the experiments.
  **date_plot_experiments_result.ipynb**: This notebook is used to plot the results of all the experiments

Before running the notebooks, it is recommended to set up a virtual environment and install dependencies from requirements.txt.

