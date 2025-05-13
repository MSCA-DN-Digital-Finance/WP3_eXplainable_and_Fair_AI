import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Define constants
TRENDS = {
    "4_1": ["upward", "downward", "periodic"],
    "4_2": ["upward_noise", "downward_noise", "periodic_noise"]
}
AGENTS = ["ddpg", "sac", "a2c"]
RUNS = [f"{i:02d}" for i in range(1, 11)]
BASE_PATH = Path("Results")

# --- Helper functions ---
def clean_allocation_column(df, col_name="allocation_weights"):
    def parse_val(val):
        if isinstance(val, str) and val.strip().startswith("["):
            val = val.strip("[]")
        return ", ".join([f"{float(x.strip()):.2f}" for x in val.split(",")])
    df[col_name] = df[col_name].apply(parse_val)
    return df

def plot_dual_line_per_episode_on_axes(df, metric1, metric2, legend1, legend2,
                                       selected_episodes, axes_row, row_idx,
                                       run_label=None, show_xticks=False, show_titles=False):
    for idx, episode in enumerate(selected_episodes):
        ax = axes_row[idx]
        df_episode = df[df["episode"] == episode]

        if df_episode.empty:
            ax.set_title(f"Ep {episode}\n(no data)", fontsize=8)
            ax.axis("off")
            continue

        df_episode = clean_allocation_column(df_episode, metric1)
        days = df_episode["day"]

        if isinstance(df_episode[metric1].iloc[0], str) and "," in df_episode[metric1].iloc[0]:
            line1 = df_episode[metric1].apply(lambda x: float(x.split(",")[0].strip()))
            line2 = df_episode[metric2].apply(lambda x: float(x.split(",")[1].strip()))
        else:
            line1 = df_episode[metric1]
            line2 = df_episode[metric2]

        ax.plot(days, line1, label=legend1, color="#1f77b4", linewidth=1.5)
        ax.plot(days, line2, label=legend2, color="#ff7f0e", linewidth=1.5)

        if show_titles:
            ax.set_title(f"Ep {episode}", fontsize=9)
        else:
            ax.set_title("")

        if show_xticks:
            ax.set_xlabel("Day")
            ax.tick_params(axis='x', labelbottom=True, rotation=45)
        else:
            ax.set_xlabel("")
            ax.tick_params(axis='x', labelbottom=False)

        if run_label and idx == 0:
            ax.set_ylabel(run_label, fontsize=9)
        else:
            ax.set_ylabel("")

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
        episodes = sorted(df["episode"].unique())
        if len(episodes) <= 5:
            selected_episodes = episodes
        else:
            mid = len(episodes) // 2
            selected_episodes = [episodes[0], episodes[(0+mid)//2], episodes[mid],
                                 (episodes[(mid+len(episodes)-1)//2]), episodes[-1]]

        plot_dual_line_per_episode_on_axes(
            df=df,
            metric1="allocation_weights",
            metric2="allocation_weights",
            legend1="Cash Weight",
            legend2="Stock Weight",
            selected_episodes=selected_episodes,
            axes_row=axes[row_idx],
            row_idx=row_idx,
            run_label=f"Run {run}",
            show_xticks=(row_idx == 9),
            show_titles=(row_idx == 0)
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1, 1), fontsize='large')
    fig.suptitle(f"Allocation Weights Over Time –  Experiment {exp[-1]} | Agent: {agent.upper()} | Trend: {trend.replace('_', ' ').title()}", fontsize=16, y=0.97)
    fig.tight_layout()
    fig.subplots_adjust(top=0.93, right=0.85)

    output_path = BASE_PATH /  f"allocation_plots_across_runs/{exp}_{agent}_{trend}_allocation_plot.png"
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