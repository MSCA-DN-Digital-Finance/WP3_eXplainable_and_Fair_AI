import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Define constants
TRENDS = {
    "4_1": ["upward", "downward", "periodic"],
    "4_2": ["upward_noise", "downward_noise", "periodic_noise"]
}
AGENTS = ["ddpg", "ppo", "a2c"]
RUNS = [f"{i:02d}" for i in range(1, 11)]
BASE_PATH = Path("Results")


def plot_10x5_grid_for_config(exp, trend, agent):
    base_dir = BASE_PATH / exp / agent / trend / "training_logs"
    fig, axes = plt.subplots(10, 5, figsize=(25, 18), sharey=True)

    for row_idx, run in enumerate(RUNS):
        file_path = base_dir / f"{run}.csv"
        if not file_path.exists():
            for col in range(5):
                axes[row_idx, col].axis("off")
            continue

        df = pd.read_csv(file_path)
        

        # Clean 'new_portfolio_value' column
        df['new_portfolio_value'] = df['new_portfolio_value'].astype(str).str.replace(",", "", regex=False).astype(float).round(2)

        episodes = sorted(df["episode"].unique())
        if len(episodes) <= 5:
            selected_episodes = episodes
        else:
            mid = len(episodes) // 2
            selected_episodes = [
                episodes[0],
                episodes[(0 + mid) // 2],
                episodes[mid],
                episodes[(mid + len(episodes) - 1) // 2],
                episodes[-1]
            ]

        for col_idx, episode in enumerate(selected_episodes):
            df_ep = df[df["episode"] == episode]
            if df_ep.empty:
                axes[row_idx, col_idx].axis("off")
                continue

            ax = axes[row_idx, col_idx]
            ax.plot(
                df_ep["day"],
                df_ep["new_portfolio_value"],
                label=f"Ep {episode}",
                color="#B7410E",
                linewidth=1.5
            )

            ax.grid(True, linestyle=":", linewidth=0.5)

            if row_idx == 9:
                ax.set_xlabel("Day", fontsize=10)
                ax.tick_params(axis='x', labelbottom=True, labelsize=8, rotation=45)
            else:
                ax.set_xlabel("")
                ax.tick_params(axis='x', labelbottom=False)

            if col_idx == 0:
                ax.set_ylabel(f"Run {run}", fontsize=10)
            else:
                ax.set_ylabel("")

            if row_idx == 0:
                ax.set_title(f"Ep {episode}", fontsize=10)

    # Add global title and legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(
        f"Portfolio Value – Exp {exp[-1]} | Agent: {agent.upper()} | Trend: {trend.replace('_', ' ').title()}",
        fontsize=16, y=0.97
    )
    fig.tight_layout(pad=1.5, h_pad=1.2, w_pad=0.8)
    fig.subplots_adjust(top=0.93)
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.99, 0.98), fontsize='large')

    output_dir = BASE_PATH / "portfolio_values_across_runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{exp}_{agent}_{trend}_portfolio_value_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()



# --- Run for all combinations ---
if __name__ == "__main__":
    for exp, trends in TRENDS.items():
        for trend in trends:
            for agent in AGENTS:
                print(f"Generating plot for: {exp} / {trend} / {agent}")
                plot_10x5_grid_for_config(exp, trend, agent)
    print("All plots generated.")
