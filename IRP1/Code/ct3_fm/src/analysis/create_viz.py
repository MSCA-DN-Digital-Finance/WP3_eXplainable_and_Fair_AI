import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import numpy as np

    # Configure publication-style typography & math rendering globally
sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",  # Use LaTeX default Computer Modern font for math
        "font.size": 11,
    }
)


# Calculate the project root (2 levels up from src/prediction/)
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)


CSV_DATASET = Path(project_root) / "artifacts" / "dataset" / "aggregated_dataset.csv"
PLOT_OUTPUT_DIR = Path(project_root) / "artifacts" / "plots"



# Label mapping dictionary using raw LaTeX strings for publication-grade formatting
LABEL_MAPPINGS = {
    # Models & Baselines
    "chronos": "Chronos-2",
    "timesfm": "TimesFM-2.5",
    "trajectory": "Trajectory",

    # Parameter Statistics (Y-Axis)
    "estimated_mean": r"$\hat{\mu}$",
    "estimated_beta": r"$\hat{\beta}$",
    "estimated_wavelength": r"$\hat{\lambda}$",
    "estimated_dwell_time": r"$\hat{\tau}$",
    "estimated_threshold": r"$\hat{\kappa}$",
    "estimated_hurst_exponent": r"$\hat{H}$",

    # Intervention Parameters (X-Axis)
    "mu": r"$\mu$",
    "beta": r"$\beta$",
    "wavelength": r"$\lambda$",
    "dwell_time": r"$\tau$",
    "threshold": r"$\kappa$",
    "hurst_exponent": r"$H$"
}

# Define explicit color palette so models always retain the exact same color
COLOR_PALETTE = {
    "Trajectory": "#7FC97F",  # Green/Teal (e.g., Trajectory)
    "TimesFM-2.5": "#FDC086",  # Orange/Salmon
    "Chronos-2": "#beaed4",  # Purple/Blue
}

def create_experiment_boxplots(
    csv_path: str, output_dir: str, include_table: bool = False
):
    """Generates structured boxplots grouped by experiment_id with consistent

    model colors, explicit legend, and an optional clean summary table underneath.
    """
    df = pd.read_csv(csv_path)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    unique_exps = sorted(df["experiment_id"].dropna().unique())
    if not unique_exps:
        print("No experiment data found to plot.")
        return

    for exp_id in unique_exps:
        df_exp = df[df["experiment_id"] == exp_id].copy()
        if df_exp.empty:
            continue

        # Map labels if mapping dict is present
        df_exp["param_stat_of"] = df_exp["param_stat_of"].map(
            lambda x: LABEL_MAPPINGS.get(x, x)
        )
        df_exp = df_exp.sort_values(by="parameter_value")

        intervention_param = df_exp["intervention_param"].iloc[0]
        param_stat_name = df_exp["param_stat_name"].iloc[0]

        x_symbol = LABEL_MAPPINGS.get(intervention_param, intervention_param)
        y_symbol = LABEL_MAPPINGS.get(param_stat_name, param_stat_name)

        # Set layout based on whether table is attached
        if include_table:
            fig, (ax, ax_table) = plt.subplots(
                2,
                1,
                figsize=(9, 7.5),
                gridspec_kw={"height_ratios": [3, 1]},
            )
        else:
            fig, ax = plt.subplots(figsize=(8, 5))

        # 1. Boxplot generation
        sns.boxplot(
            data=df_exp,
            x="parameter_value",
            y="param_stat_value",
            hue="param_stat_of",
            ax=ax,
            palette=COLOR_PALETTE,
            linewidth=1.2,
            boxprops=dict(alpha=0.85),
        )

        # Optional: Add ground truth reference line y = x if intervention matches metric
        # ax.plot(ax.get_xticks(), ax.get_xticks(), color="gray", linestyle="--", alpha=0.6, label="Ideal")

        ax.set_xlabel(
            f"Intervention Parameter ({x_symbol})", fontsize=11, labelpad=8
        )
        ax.set_ylabel(
            f"Parameter Statistic ({y_symbol})", fontsize=11, labelpad=8
        )
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)

        # 2. Fixed Legend Placement (Top-Left inside plot)
        ax.legend(
            loc="upper left",
            frameon=True,
            facecolor="white",
            edgecolor="none",
            fontsize=9.5,
        )

        # 3. Optional Empirical Table Underneath
        if include_table:
            # Group stats across entire experiment for each model/source
            stats = (
                df_exp.groupby("param_stat_of")["param_stat_value"]
                .agg(
                    Min="min",
                    Q25=lambda x: x.quantile(0.25),
                    Median="median",
                    Mean="mean",
                    Q75=lambda x: x.quantile(0.75),
                    Max="max",
                )
                .round(4)
            )

            ax_table.axis("off")
            table_data = stats.reset_index().values
            col_labels = ["Source", "Min", "25%", "Median", "Mean", "75%", "Max"]

            tab = ax_table.table(
                cellText=table_data,
                colLabels=col_labels,
                cellLoc="center",
                loc="center",
            )
            tab.auto_set_font_size(False)
            tab.set_fontsize(9)
            tab.scale(1.0, 1.3)

        plt.tight_layout()

        file_name = out_path / f"experiment_{exp_id}_boxplot.png"
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f" -> Saved plot to: {file_name}")


def save_individual_scatter_plots(csv_path: str, output_dir: str):
    """Pivots the dataset to align foundation model outputs against the ground-truth

    trajectory values on a 1-to-1 scatter plot grid.
    """
    # 1. Load the dataset
    df = pd.read_csv(csv_path)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    unique_exps = sorted(df["experiment_id"].unique())

    if not unique_exps:
        print("No experiment data found to plot.")
        return

    print(f"Found experiments: {unique_exps}. Generating scatter plots...")

    # 2. Process each experiment individually
    for exp_id in unique_exps:
        df_exp = df[df["experiment_id"] == exp_id].copy()

        if df_exp.empty:
            continue

        # Extract metadata attributes for mapped labels
        param_stat_name = df_exp["param_stat_name"].iloc[0]
        generator_name = df_exp["generator_name"].iloc[0]

        # Fetch mapped symbols
        y_symbol = LABEL_MAPPINGS.get(param_stat_name, param_stat_name)
        gen_label = LABEL_MAPPINGS.get(generator_name, generator_name)

        # 3. Pivot the table so each source has its own column per run_id
        pivot_df = df_exp.pivot(
            index=["run_id", "parameter_value"],
            columns="param_stat_of",
            values="param_stat_value",
        ).reset_index()

        # Safety Check: Verify trajectory baseline data is present
        if "trajectory" not in pivot_df.columns:
            print(
                f"Skipping Exp {exp_id}: No 'trajectory' baseline values found."
            )
            continue

        # Identify model columns
        model_cols = [
            col
            for col in pivot_df.columns
            if col not in ["run_id", "parameter_value", "trajectory"]
        ]

        if not model_cols:
            print(
                f"Skipping Exp {exp_id}: No evaluation models found to contrast against trajectory."
            )
            continue

        # 4. Melt the pivoted data back down *excluding* trajectory for Seaborn hue
        melted_df = pivot_df.melt(
            id_vars=["run_id", "parameter_value", "trajectory"],
            value_vars=model_cols,
            var_name="model_name",
            value_name="model_param_stat_value",
        )

        # Map internal model strings to publication labels (e.g., chronos -> Chronos-2)
        melted_df["model_label"] = melted_df["model_name"].map(
            lambda x: LABEL_MAPPINGS.get(x, x)
        )

        # 5. Initialize the figure
        fig, ax = plt.subplots(figsize=(6.5, 6))

        # Create the scatter plot using mapped model names
        sns.scatterplot(
            data=melted_df,
            x="trajectory",
            y="model_param_stat_value",
            hue="model_label",
            style="model_label",  # Distinct shapes per model
            palette="Set1",
            s=60,
            alpha=0.8,
            ax=ax,
        )

        # 6. Add a diagonal 1-to-1 reference identity line (y = x)
        all_vals = pd.concat(
            [melted_df["trajectory"], melted_df["model_param_stat_value"]]
        ).dropna()
        if not all_vals.empty:
            min_val, max_val = all_vals.min(), all_vals.max()
            padding = (
                (max_val - min_val) * 0.08 if max_val != min_val else 0.1
            )
            lims = [min_val - padding, max_val + padding]

            ax.plot(
                lims,
                lims,
                color="darkgray",
                linestyle="--",
                linewidth=1.5,
                label=r"Ideal ($y=x$)",
            )
            ax.set_xlim(lims)
            ax.set_ylim(lims)

        # Clean LaTeX labels and titles
        ax.set_xlabel(
            f"Ground Truth Trajectory Statistic ({y_symbol})", fontsize=10
        )
        ax.set_ylabel(
            f"Forecasted Parameter Statistic ({y_symbol})", fontsize=10
        )

        # Refresh legend with mapped names and ideal line
        ax.legend(title=None, loc="best", frameon=True)
        ax.set_aspect("equal", adjustable="box")  # Maintain square 1-to-1 grid

        plt.tight_layout()

        # Save image asset
        file_name = out_path / f"experiment_{exp_id}_scatter_consistency.png"
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f" -> Saved consistency scatter plot to: {file_name}")



def plot_experiment_trajectories(csv_path: str, output_dir: str):
    """
    Generates a dynamically stacked plot (N+1 x 1 grid) per experiment.
    - Top Plot: The FIRST ground-truth trajectory per parameter_value.
    - Subsequent Plots: Corresponding predicted trajectories found inside 
      prediction/{run_id}/{model_name}/predictions.npz.
    """
    df = pd.read_csv(csv_path)
    
    # Establish root path relative to the CSV file location
    artifacts_dir = Path(csv_path).parent.parent
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    unique_exps = sorted(df['experiment_id'].unique())
    if not unique_exps:
        print("No experiment data found.")
        return

    print(f"Generating dynamically stacked trajectory plots for experiments: {unique_exps}...")
    sns.set_theme(style="whitegrid")
    
    for exp_id in unique_exps:
        df_exp = df[df['experiment_id'] == exp_id].copy()
        
        # Metadata recovery fallback strings
        intervention_param = df_exp['intervention_param'].iloc[0] if len(df_exp) > 0 else "Parameter"
        generator_name = df_exp['generator_name'].iloc[0] if len(df_exp) > 0 else ""
        param_values = sorted(df_exp['parameter_value'].unique())
        
        # Step 1: Collect first run_ids and inspect prediction subdirectories to locate models
        model_paths = {}  # Format: { model_name: { param_value: file_path } }
        gt_paths = {}     # Format: { param_value: file_path }
        
        for val in param_values:
            df_val = df_exp[df_exp['parameter_value'] == val]
            run_id = df_val['run_id'].iloc[0] if len(df_val) > 0 else None
            if not run_id:
                continue
                
            # Track ground-truth file
            gt_file = artifacts_dir / "generation" / str(run_id) / "trajectory.npz"
            if gt_file.exists():
                gt_paths[val] = gt_file
                
            # Scan prediction/{run_id} subfolders dynamically
            pred_base_dir = artifacts_dir / "prediction" / str(run_id)
            if pred_base_dir.exists():
                for model_dir in pred_base_dir.iterdir():
                    if model_dir.is_dir():
                        pred_file = model_dir / "predictions.npz"
                        if pred_file.exists():
                            model_name = model_dir.name
                            if model_name not in model_paths:
                                model_paths[model_name] = {}
                            model_paths[model_name][val] = pred_file

        # Identify all unique models discovered across this experiment
        discovered_models = sorted(list(model_paths.keys()))
        num_subplots = 1 + len(discovered_models)  # 1 (Ground Truth) + N models
        
        if not gt_paths and not model_paths:
            print(f"Skipping Exp {exp_id}: No array (.npz) files found on disk.")
            continue

        # Step 2: Initialize an (N+1) x 1 grid layout
        fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 3.5 * num_subplots), sharex=True, sharey=True)
        
        # Force axes into a sequence list even if it's a single subplot layout
        if num_subplots == 1:
            axes = [axes]
            
        # Helper sequence to safely read multi-dimensional npz structures
        def load_npz_sequence(path):
            with np.load(path) as data:
                # If 'x' is in the npz file (ground truth target), use it. 
                # Otherwise, fall back to the first available array key (like 'predictions').
                key = "x" if "x" in data.files else data.files[0]
                arr = data[key]
                return arr.flatten() if arr.ndim > 1 else arr

        # Step 3: Draw Ground Truth on the Top Subplot (Index 0)
        ax_top = axes[0]
        for val, path in gt_paths.items():
            try:
                y_vals = load_npz_sequence(path)
                ax_top.plot(y_vals, label=f"{intervention_param} = {val}", linewidth=2, marker='o', markersize=3)
            except Exception as e:
                print(f"Error reading GT array for param {val}: {e}")
        ax_top.set_title("Trajectory", fontsize=11, weight='bold')
        ax_top.legend(loc='upper right', frameon=True, fontsize=9)
        ax_top.set_ylabel("Value Magnitude")

        # Step 4: Iteratively build subplots for each dynamically discovered model
        for idx, model_name in enumerate(discovered_models, start=1):
            ax_model = axes[idx]
            model_configs = model_paths[model_name]
            
            for val, path in model_configs.items():
                try:
                    y_vals = load_npz_sequence(path)
                    ax_model.plot(y_vals, label=f"{LABEL_MAPPINGS.get(intervention_param, intervention_param)} = {val}", linewidth=2, linestyle='--')
                except Exception as e:
                    print(f"Error reading {model_name} array for param {val}: {e}")
            
            ax_model.set_title(LABEL_MAPPINGS.get(model_name, model_name), fontsize=11, weight='bold')
            ax_model.legend(loc='upper right', frameon=True, fontsize=9)
            ax_model.set_ylabel("Value Magnitude")
            
        # Add shared X labels on the very bottom axis frame
        axes[-1].set_xlabel("Timestep", fontsize=11)
        
        plt.tight_layout()
        
        # Save structural visualization image
        file_name = out_path / f"experiment_{exp_id}_trajectory_cascade.png"
        plt.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f" -> Saved stacked trajectory cascade to: {file_name}")

        

if __name__ == "__main__":
    # Adjust file paths to match your directory structure

    
    create_experiment_boxplots(CSV_DATASET, PLOT_OUTPUT_DIR)
    save_individual_scatter_plots(CSV_DATASET, PLOT_OUTPUT_DIR)
    plot_experiment_trajectories(CSV_DATASET, PLOT_OUTPUT_DIR )
