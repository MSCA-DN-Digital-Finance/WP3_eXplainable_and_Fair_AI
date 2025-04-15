# Reinforcement Learning Project: Portfolio Management

This repository implements and experiments with reinforcement learning (RL) strategies for portfolio management. It is structured to facilitate modular development and testing of various RL agents and market environments.

## Repository Structure

- **`agents.py`**: Implements different RL agents, such as Q-Learning and baseline agents, which interact with the environment to learn optimal portfolio management strategies.

- **`environments.py`**: Defines the market environments in which the agents operate, including the dynamics of asset prices and portfolio valuations.

- **`generators.py`**: Implements various data generators that simulate market conditions, providing the environments with synthetic data for training and testing the agents.

-  **`plot_functions.py`**: Defines various functions that are used to plot the results of the experiments

- **Notebooks**:  These Jupyter notebooks are used to run experiments, analyze results, and compare the performance of different agents and strategies under various market scenarios.
  
    -> **`date_train_experiments.ipynb`**: This notebook is used to train all the experiments.
  
    -> **`date_plot_experiments_result.ipynb`**: This notebook is used to plot the results of all the experiments

Before running the notebooks, it is recommended to set up a virtual environment and install dependencies from requirements.txt.


## Price Generators

This module provides a collection of price generators for simulating different market behaviors. These can be used for testing trading algorithms, simulating time series data, or creating synthetic datasets.

---

### 1. `NormalPriceGenerator`
Generates prices sampled from a normal distribution.

- **Use case**: Purely random prices, e.g., for baseline testing.
- **Parameters**:
  - `mu`: Mean of the normal distribution.
  - `sigma`: Standard deviation of the distribution.
- **Method**:
  - `generate_price()`: Returns a new price from `N(mu, sigma)`.

---

### 2. `LinearTrendPriceGenerator`
Creates a linearly increasing or decreasing price series.

- **Use case**: Simulate a basic trend (upward or downward).
- **Parameters**:
  - `start_price`: Starting price (default 10).
  - `up`: Boolean flag to control trend direction.
- **Method**:
  - `generate_price()`: Returns next price, incremented or decremented by 1.

---

### 3. `CashPriceGenerator`
Simulates a flat, unchanging price (e.g., cash holdings).

- **Use case**: Represent assets with zero volatility or fixed return.
- **Method**:
  - `generate_price(last_price)`: Returns the same price as before.

---

### 4. `NoisyTrendPriceGenerator`
Generates prices with a noisy exponential trend using log-normal returns.

- **Use case**: Simulate realistic price series with drift and volatility.
- **Parameters**:
  - `mean`: Mean of log return distribution.
  - `variance`: Variance of log return distribution.
- **Method**:
  - `generate_price(last_price)`: Returns the next price using `log_return ~ N(mean, variance)`.

---

### 5. `PeriodicTrendPriceGenerator`
Generates a clean sine wave with a configurable base value, amplitude, and frequency.

- **Use case**: Simulate cyclical market behavior without noise.
- **Parameters**:
  - `start`: Base value (e.g., long-term average).
  - `amplitude`: Height of sine wave.
  - `frequency`: Controls wave oscillation speed (radians per step).
- **Method**:
  - `generate()`: Returns next value from sine wave.

---

### 6. `NoisyPeriodicTrendPriceGenerator`
Extends the periodic generator by adding Gaussian noise to the sine wave.

- **Use case**: Simulate cyclical behavior with added market noise.
- **Parameters**:
  - `start`: Base value (e.g., long-term average).
  - `amplitude`: Height of sine wave.
  - `frequency`: Oscillation speed (radians per step).
  - `mu`: Mean of noise.
  - `sigma`: Standard deviation of noise.
- **Method**:
  - `generate()`: Returns noisy sine-based price.


