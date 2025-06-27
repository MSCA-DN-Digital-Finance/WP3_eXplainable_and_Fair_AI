# Reinforcement‑Learning Project Group 6

This project investigates whether state-of-the-art reinforcement learning (RL) agents can effectively detect and exploit patterns in time series data, such as trends and cycles, when trained in controlled environments. By training RL agents on various synthetic market scenarios, we evaluate their ability to learn optimal asset allocation strategies for a simple two-asset portfolio (stock + cash). The results serve as empirical evidence to possibly falsify the hypotheses proposed in our accompanying paper.

Every experiment is self‑contained—it ships the exact generators, environment definition, configuration, and the results it produces—so you can reproduce or extend any part of the study without touching the rest.

N.B.: Experiment 4_1 and 4_2 correspond to Experiment 1 and Experiment 2 in our paper, while Experimentation MJ refers to Experiment 3. 

---

## Repository layout

```text
rl_project_group_6/
├── experiments/
│   ├── experiment_1/           # single‑asset prototype (sparse vs dense reward)
│   ├── experiment_4/           # FinRL agents + parallel orchestrator (90 runs)
│   ├── experimentation_mj/     # Experimentation to develop algorithms for experiment 3 (in paper)
│   └── first_implementations/  # early notebooks: scratch Q‑table agent, scratch DQN, etc.
├── requirements.txt
└── README.md  ← you are here

```

**Key files inside `experiments/experiment_4/` and `experiments/experimentation_mj/` (with some minor changes) :**

| File / folder      | Purpose                                                    |
| ------------------ | ---------------------------------------------------------- |
| `base_trainer.py`  | Train **one** run (parameterised via CLI flags)            |
| `orchestrator.py`  | Launch every (trend, agent, run) combo in parallel         |
| `generators.py`    | Price‑series generators used *only* by Experiment 4        |
| `environments.py`  | OpenAI Gym‑style portfolio environment for Experiment 4    |
| `plot_functions.py` | Helper plots (returns, allocations, rewards) |
| `generate_timeseries.py` | Generate synthetic price **CSV & PNG** files under `Data/` |
| `generate_allocation_plots_across_runs.py` | It generates 10×5 grid plots of allocation weights over episodes for each (experiment, trend, agent) configuration, saving one plot per config |
| `generate_portfolio_values_plots_across_runs.py` | It generates 10×5 grid plots of portfolio values over episodes for each (experiment, trend, agent) configuration, saving one plot per config |
| `calculate_evaluation_metrics.ipynb` | Computes the average allocation error and the difference (ΔV) between the agent's final portfolio value and the optimal portfolio value across multiple runs |
| `Data/`            | Synthetic price CSVs + PNG previews for each trend         |
| `Results/`         | CSV logs and PNG plots produced by each run                |

---

## Environment setup

```bash
conda create -n rl-group-6
conda activate rl-group-6
pip install -r requirements.txt
```

---

## Price‑series generators

| Class | Behaviour | Experiment |
|-------|-----------|------------------|
| `NormalPriceGenerator`             | i.i.d. draws from **N(μ, σ)** | `first_implementations` (250306_hold_vs_normal.ipynb notebook; 250306_qlearning_vs_normal.ipynb notebook) |
| `LinearPriceGenerator`             | Strict **+1** increment per step (monotonic ↑) | `first_implementations` (250306_hold_vs_linear.ipynb notebook;  250306_qlearning_vs_linear.ipynb notebook) |
| `LinearTrendPriceGenerator`        | Deterministic ↑ or ↓ trend (±1 each step) | `experiment_4` (4.1) & `experimentation_mj` (250619_hc_binary_udp)|
| `PeriodicTrendPriceGenerator`      | Clean sine wave               | `experiment_4` (4.2) & `experimentation_mj` (250619_hc_binary_udp)|
| `NoisyTrendPriceGenerator`         | Geometric Brownian motion     | `experiment_4` (4.1) & `experimentation_mj` (250619_hc_binary_udp_noise)|
| `NoisyPeriodicTrendPriceGenerator` | Sine wave + Gaussian noise    | `experiment_4` (4.2) & `experimentation_mj` (250619_hc_binary_udp_noise)|
| `CashPriceGenerator`               | Flat price (zero volatility)  | all experiments |

Each experiment imports only the generators it needs.

## Portfolio environments  

The **environments.py** module in each experiment folder converts price
series into an RL‑ready *state → action → reward* loop.  We ship three
variants, each tuned for its experiment’s scope:

| Class / module | Main features | Used in |
|----------------|---------------|---------|
| `PortfolioEnvironment` | **Toy loop** for Q‑table / DQN prototypes.<br> • 1 stock + cash<br> • Action ∈ {buy, sell, hold} translated into ±10 % allocation steps<br> • Reward = Δ portfolio − Tx cost | `experiments/first_implementations` |
| `StockPortfolioEnv`<br>(experiment 1) | **Gym‑compatible** env (Stable‑Baselines 3).<br> • State = current price (t) + 5 past returns<br> • Two reward modes: *sparse* vs *dense* Tx‑cost<br> • Writes per‑timestep CSV/PNG for analysis | `experiments/experiment_1` |
| `StockPortfolioEnv`<br>(experiment 4) | **Gym‑compatible** env (Stable‑Baselines 3)<br> • State = current price only (lag features removed)<br> • Reward = Δ portfolio value (no Tx cost) | `experiments/experiment_4` |
| `StockPortfolioEnvBinary` |**Gym‑compatible** env (Stable‑Baselines 3)<br> • Action ∈ [0, 1]² → mapped to either cash or asset<br> • State = engineered features: past returns, MAs, volatility, etc.<br> • Reward = price change × position <br> • Includes logging & episode tracking | `experiments/experimentation_mj` |


*`StockPriceSimulator`* is a helper class (found in both experiment 1 and 4)
that turns price generators into long‑form DataFrames which feed these
environments.

---

## Experiment 1 — single asset, sparse vs dense reward

| Item                | Setting                                                   |
| ------------------- | --------------------------------------------------------- |
| Portfolio start     | **\$1 000** (50 % stock, 50 % cash)                       |
| State               | price (t) + five lags (t‑1…t‑5)                           |
| Price generator     | Stock: price\_next = price \* exp(N(0.1, 2))Cash: N(0, 0) |
| Agent               | **DDPG** (Stable‑Baselines 3 defaults)                    |
| Episodes            | 5 × 500 steps                                             |
| Reward A (*sparse*) | –1 % Tx cost each step; terminal Δ portfolio              |
| Reward B (*dense*)  | Δ portfolio –1 % Tx cost each step                        |
| Outputs             | reward, allocation, portfolio value per step              |

The experiment 1 is trained in the notebook 250318_train_experiment_1.ipynb and the notebook 250318_plot_experiment_1 is used to produce the plots which are then stored under experiments/experiment\_1/Results/.

---

## Experiment 4

### 4 .1 (no noise)

| Setting | Value                                   |
| ------- | --------------------------------------- |
| Trends  | upward / downward / periodic            |
| Agents  | **DDPG**, **SAC**, **A2C**              |
| Runs    | 10 per (trend, agent) → 90 runs         |
| Episode | 500 episodes × 1 000 steps              |
| Reward  | FinRL NeurIPS 2020 default (no Tx cost) |
| Logging | reward, allocation, portfolio value     |

Outputs → Results/4\_1/\<agent>/\<trend>/.

### 4 .2 (noisy)

Identical to 4 .1 but every generator adds Gaussian noise (see generators.py).\
Outputs → Results/4\_2/\<agent>/\<trend>/.

---

### Running the experiments 4.1 and 4.2

#### Run everything in parallel (experiment 4)

```bash
cd experiments/experiment_4
python orchestrator.py -j 4   # limit workers with -j N
```

#### Run a single job (debug)

```bash
cd experiments/experiment_4
python base_trainer.py \
    --experiment 4_1 \
    --trend upward \
    --agent ddpg \
    --run 01 \
    --episodes 500 \
    --days 1000
# results → Results/4_1/ddpg/upward/
```

## Experimentation MJ

Parallel training of RL agents on synthetic trends with **binary position-based reward** logic (cash vs. asset). Uses `250619_hc_binary_udp_orchestrator.py` to run all jobs as isolated processes with detailed logging and failure handling.


| Setting     | Value                                                                 |
|-------------|-----------------------------------------------------------------------|
| Experiments | `250619_hc_binary_udp`, `250619_hc_binary_udp_noise`                 |
| Trends      | `upward`, `downward`, `periodic`, `upward_noise`, `downward_noise`, `periodic_noise` |
| Agents      | **A2C**, **PPO**, **DDPG**                                            |
| Runs        | 10 per (experiment, trend, agent) → 180 total runs                   |
| Episodes    | 50 per run                                                            |
| Reward      | Reward = Δ price × position, Position = cash or asset based on action |
| Logging     | reward, allocation, portfolio value        |


### Run everything in parallel

```bash
cd experiments/experimentation_mj
python 250619_hc_binary_udp_orchestrator.py -j 4  # use -j N to control parallelism
```

---

## Acknowledgments
Acknowledgments
The goal of this project is to test whether state-of-the-art implementations, such as the FinRL agents, can reliably learn optimal exploitation strategies in synthetic market environments. To achieve this, the project builds upon the FinRL framework, specifically leveraging the StockPortfolioEnv and hyperparameters from the FinRL portfolio allocation NeurIPS 2020 workshop notebook (https://github.com/timqqt/FinRL-Library/FinRL_portfolio_allocation_NeurIPS_2020.ipynb). This implementation builds the environment for portfolio management allocation and uses Stable Baselines 3 algorithms to train RL agents for optimal asset allocation. 


Funded by the European Union. Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or European Research Executive Agency (REA). Neither the European Union nor the granting authority can be held responsible for them.

![EU Logo](images/eu_funded_logo.jpg)

## License

MIT — see LICENSE.