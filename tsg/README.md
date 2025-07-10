# tsg – Modular Time Series Generator Library

**tsg** is a lightweight Python library for generating synthetic time series data using modular and composable generators.  
It is designed for research, simulation, and testing of sequential decision-making algorithms, especially in finance and machine learning.

---

## 🚀 Features

- 📈 Core generators: linear trend, periodic trend, constant value
- 🎲 Noise wrappers: add Gaussian noise or build your own modifiers
- 🔁 Stateful generators with `reset()` support
- 🧱 Easy to extend with your own generator or modifier classes

---

## 📦 Installation

### 🔧 Option 1: Install Locally (Editable Mode)

Clone the repository and install in editable mode:

```bash
git clone https://github.com/your-username/tsg.git
cd tsg
pip install -e .
```
## 🛠️ Example Usage

Here’s how to generate a noisy linear trend time series using `tsg`:

```python
from tsg.generators import LinearTrendGenerator
from tsg.modifiers import GaussianNoise

# Create a linear trend generator (increasing by +1 each step)
base_generator = LinearTrendGenerator(start_price=100, up=True)

# Wrap it with Gaussian noise (mean=0, std=1)
noisy_generator = GaussianNoise(base_generator, mu=0.0, sigma=1.0)

# Generate a few data points
prices = []
for _ in range(10):
    price = noisy_generator.generate_price(None)
    prices.append(price)

print(prices)
```

## 🧠 API Overview

### Core Generators (`tsg.generators`)

| Class                    | Description                                              |
|--------------------------|----------------------------------------------------------|
| `LinearTrendGenerator`   | Linearly increases or decreases the price at each step   |
| `ConstantGenerator`      | Returns a fixed price (e.g., simulates cash)             |
| `PeriodicTrendGenerator` | Generates a sinusoidal time series with set amplitude and frequency |

### Modifier Wrappers (`tsg.modifiers`)

| Class          | Description                                                  |
|----------------|--------------------------------------------------------------|
| `GaussianNoise`| Adds Gaussian noise (`N(mu, sigma)`) to any base generator   |

All components implement the `BaseGenerator` interface with:

- `generate_price(last_price)` – returns the next value in the sequence
- `reset()` – resets any internal state (optional for stateless generators)


---

## Acknowledgments
Acknowledgments
The goal of this project is to test whether state-of-the-art implementations, such as the FinRL agents, can reliably learn optimal exploitation strategies in synthetic market environments. To achieve this, the project builds upon the FinRL framework, specifically leveraging the StockPortfolioEnv and hyperparameters from the FinRL portfolio allocation NeurIPS 2020 workshop notebook (https://github.com/timqqt/FinRL-Library/FinRL_portfolio_allocation_NeurIPS_2020.ipynb). This implementation builds the environment for portfolio management allocation and uses Stable Baselines 3 algorithms to train RL agents for optimal asset allocation. 


Funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or European Research Executive Agency (REA). Neither the European Union nor the granting authority can be held responsible for them.

![EU Logo](images/eu_funded_logo.jpg)

## License

MIT — see LICENSE.