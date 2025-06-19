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
from environments import StockPriceSimulatorExtended

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTDIR = Path("Data")
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
        {"STOCK": 1000, "CASH": 1000},
        {
            "STOCK": LinearTrendPriceGenerator(start_price=1000, up=False),
            "CASH": CashPriceGenerator(),
        },
    ),
    (
        "periodic",
        {"STOCK": 500, "CASH": 500},
        {
            "STOCK": PeriodicTrendPriceGenerator(start=500, amplitude=50, frequency=0.15),
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
        {"STOCK": 500, "CASH": 500},
        {
            "STOCK": NoisyPeriodicTrendPriceGenerator(
                start=500, 
                amplitude=50, 
                frequency=0.15,
                mu=0,
                sigma=10,
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
        sim = StockPriceSimulatorExtended(days=days, initial_prices=initial_prices, generators=generators)
        df = sim.generate_prices()

        # Save PNG plot
        plot_prices(df, OUTDIR / f"{name}.png")

        # Save raw time‑series to CSV
        csv_path = OUTDIR / f"{name}.csv"
        df.to_csv(csv_path, index=False)

        print(f"✓ {name}.png & {name}.csv saved")


if __name__ == "__main__":
    main()