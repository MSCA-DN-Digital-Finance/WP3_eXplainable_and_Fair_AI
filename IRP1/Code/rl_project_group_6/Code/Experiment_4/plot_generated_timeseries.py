from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

from generators import (
    CashPriceGenerator,
    LinearTrendPriceGenerator,
    PeriodicTrendPriceGenerator,
    NoisyTrendPriceGenerator,
    NoisyPeriodicTrendPriceGenerator,
)
from environments import StockPriceSimulator

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTDIR = Path("Results/Experiment_4/Timeseries_plots/")
OUTDIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = [
    # name,       initial_prices,                        generators
    (
        "upward",
        {"STOCK": 100, "CASH": 100},
        {
            "STOCK": LinearTrendPriceGenerator(start_price=100, up=True),
            "CASH": CashPriceGenerator(),
        },
    ),
    (
        "downward",
        {"STOCK": 1005, "CASH": 1000},
        {
            "STOCK": LinearTrendPriceGenerator(start_price=1005, up=False),
            "CASH": CashPriceGenerator(),
        },
    ),
    (
        "periodic",
        {"STOCK": 100, "CASH": 100},
        {
            "STOCK": PeriodicTrendPriceGenerator(start=100, amplitude=50, frequency=0.15),
            "CASH": CashPriceGenerator(),
        },
    ),
    (
        "upward_noise",
        {"STOCK": 100, "CASH": 100},
        {
            "STOCK": NoisyTrendPriceGenerator(mean=0.001, variance=0.02),
            "CASH": CashPriceGenerator(),
        },
    ),
    (
        "downward_noise",
        {"STOCK": 1000, "CASH": 1000},
        {
            "STOCK": NoisyTrendPriceGenerator(mean=-0.001, variance=0.02),
            "CASH": CashPriceGenerator(),
        },
    ),
    (
        "periodic_noise",
        {"STOCK": 100, "CASH": 100},
        {
            "STOCK": NoisyPeriodicTrendPriceGenerator(
                start=10.0,
                amplitude=5.0,
                frequency=0.2,
                mu=0.0,
                sigma=0.7,
            ),
            "CASH": CashPriceGenerator(),
        },
    ),
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def plot_prices(df, save_path: Path) -> None:
    plt.figure(figsize=(12, 6))
    for tic in df["tic"].unique():
        if tic == "CASH":
            continue
        sub = df[df["tic"] == tic]
        plt.plot(sub["date"], sub["close"], label=tic)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(save_path.stem.replace("_", " ").title())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main(days: int = 1000):
    for name, initial_prices, generators in SCENARIOS:
        sim = StockPriceSimulator(days=days, initial_prices=initial_prices, generators=generators)
        df = sim.generate_prices()
        plot_prices(df, OUTDIR / f"{name}.png")
        print(f"✓ {name}.png saved")


if __name__ == "__main__":
    main()