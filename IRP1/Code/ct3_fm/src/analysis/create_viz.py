import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys


# Calculate the project root (2 levels up from src/prediction/)
project_root = str(Path(__file__).resolve().parents[2])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils import load_config

CSV_DATASET = Path(project_root) / "artifacts" / "dataset" / "aggregated_dataset.csv"
PLOT_OUTPUT_DIR = Path(project_root) / "artifacts" / "plots"


def create_experiment_boxplots(csv_path: str, output_dir: str):
    """
    Generates structured boxplots grouped by experiment_id, with parameter values 
    on the x-axis, metrics on the y-axis, and separated by the model/trajectory source.
    """
    # 1. Load the dataset
    df = pd.read_csv(csv_path)
    
    # Ensure our save folder exists
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Identify unique experiments present in data
    unique_exps = sorted(df['experiment_id'].unique())
    num_experiments = len(unique_exps)
    
    if num_experiments == 0:
        print("No experiment data found to plot.")
        return
   
    print(f"Found experiments: {unique_exps}. Generating box plots...")
    
    # Set a clean visual theme for all plots
    sns.set_theme(style="whitegrid")
    
    # 2. Loop through each experiment and generate a dedicated file
    for exp_id in unique_exps:
        # Isolate rows belonging to the current experiment
        df_exp = df[df['experiment_id'] == exp_id].copy()
        
        # Sort the dataframe by parameter values so the x-axis trends logically
        df_exp = df_exp.sort_values(by='parameter_value')
        
        # Dynamically fetch metric and parameter names for titles/labels
        intervention_param = df_exp['intervention_param'].iloc[0] if len(df_exp) > 0 else "parameter"
        param_stat_name = df_exp['param_stat_name'].iloc[0] if len(df_exp) > 0 else "value"
        generator_name = df_exp['generator_name'].iloc[0] if len(df_exp) > 0 else ""
        
        # Initialize a new, dedicated figure window for this specific plot
        fig, ax = plt.subplots(figsize=(8, 5.5))
        
        # Draw the boxplot grouped by 'param_stat_of'
        sns.boxplot(
            data=df_exp,
            x='parameter_value',
            y='param_stat_value',
            hue='param_stat_of',
            ax=ax,
            palette='Set2',  # Clean color scheme (chronos, timesfm, trajectory)
            linewidth=1.5
        )
        
        # Add customized typography and labels
        ax.set_title(
            f"Experiment {exp_id} ({generator_name})\n{param_stat_name} vs {intervention_param}", 
            fontsize=13, 
            pad=12, 
            weight='bold'
        )
        ax.set_xlabel(f"Intervention Parameter: {intervention_param}", fontsize=11, labelpad=8)
        ax.set_ylabel(param_stat_name, fontsize=11, labelpad=8)
        
        # Clean legend configuration
        ax.legend(title="Source System", loc='best', frameon=True)
        
        # Tighten elements so text labels aren't cut off at the margins
        plt.tight_layout()
        
        # 3. Save as a standalone image file
        file_name = out_path / f"experiment_{exp_id}_boxplot.png"
        plt.savefig(file_name, dpi=300)  # 300 DPI ensures crisp resolution for documents
        plt.close(fig)                   # Free memory explicitly
        
        print(f" -> Saved standalone plot to: {file_name}")



def save_individual_scatter_plots(csv_path: str, output_dir: str):
    """
    Pivots the dataset to align foundation model outputs against the 
    ground-truth trajectory values on a 1-to-1 scatter plot grid.
    """
    # 1. Load the dataset
    df = pd.read_csv(csv_path)
    
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    unique_exps = sorted(df['experiment_id'].unique())
    
    if not unique_exps:
        print("No experiment data found to plot.")
        return

    print(f"Found experiments: {unique_exps}. Generating scatter plots...")
    sns.set_theme(style="whitegrid")
    
    # 2. Process each experiment individually
    for exp_id in unique_exps:
        df_exp = df[df['experiment_id'] == exp_id].copy()
        
        # Extract metadata attributes for labels
        intervention_param = df_exp['intervention_param'].iloc[0] if len(df_exp) > 0 else "parameter"
        param_stat_name = df_exp['param_stat_name'].iloc[0] if len(df_exp) > 0 else "value"
        generator_name = df_exp['generator_name'].iloc[0] if len(df_exp) > 0 else ""
        
        # 3. Pivot the table so each source has its own column per run_id
        # This transforms rows of (chronos, timesfm, trajectory) into separate parallel columns
        pivot_df = df_exp.pivot(
            index=["run_id", "parameter_value"], 
            columns="param_stat_of", 
            values="param_stat_value"
        ).reset_index()
        
        # Safety Check: Verify that we actually have trajectory data and model data to plot
        if 'trajectory' not in pivot_df.columns:
            print(f"Skipping Exp {exp_id}: No 'trajectory' baseline values found.")
            continue
            
        # Identify the model columns (anything that isn't run_id, parameter_value, or trajectory)
        model_cols = [col for col in pivot_df.columns if col not in ['run_id', 'parameter_value', 'trajectory']]
        
        if not model_cols:
            print(f"Skipping Exp {exp_id}: No evaluation models found to contrast against trajectory.")
            continue
            
        # 4. Melt the pivoted data back down *excluding* trajectory, so we can use seaborn's hue
        melted_df = pivot_df.melt(
            id_vars=["run_id", "parameter_value", "trajectory"],
            value_vars=model_cols,
            var_name="model_name",
            value_name="model_param_stat_value"
        )
        
        # 5. Initialize the figure
        fig, ax = plt.subplots(figsize=(6.5, 6))
        
        # Create the scatter plot
        sns.scatterplot(
            data=melted_df,
            x="trajectory",
            y="model_param_stat_value",
            hue="model_name",
            style="model_name",   # Gives different shapes to different models
            palette="Set1",
            s=60,                 # Size of marker dots
            alpha=0.8,
            ax=ax
        )
        
        # 6. Add a diagonal 1-to-1 reference identity line (y = x)
        # We find the min/max limits based on the actual ground truth distribution
        all_vals = pd.concat([melted_df['trajectory'], melted_df['model_param_stat_value']]).dropna()
        if not all_vals.empty:
            min_val, max_val = all_vals.min(), all_vals.max()
            # Add padding to limits so dots don't sit right on the border
            padding = (max_val - min_val) * 0.08 if max_val != min_val else 0.1
            lims = [min_val - padding, max_val + padding]
            
            ax.plot(lims, lims, color='darkgray', linestyle='--', linewidth=1.5)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
        
        # Set titles and formatting details
        ax.set_title(
            f"Experiment {exp_id} ({generator_name}): Model vs Trajectory Consistency\nMetric: {param_stat_name}", 
            fontsize=11, 
            pad=12, 
            weight='bold'
        )
        ax.set_xlabel(f"Trajectory Parameter Statistic", fontsize=10)
        ax.set_ylabel(f"Foundation Model Forecast Parameter Statistic", fontsize=10)
        
        # Refresh the legend to show both the models and the identity line reference safely
        ax.legend(loc='best', frameon=True)
        ax.set_aspect('equal', adjustable='box') # Keeps the square shape invariant for clear y=x viewing
        
        plt.tight_layout()
        
        # Save out the unique standalone image asset
        file_name = out_path / f"experiment_{exp_id}_scatter_consistency.png"
        plt.savefig(file_name, dpi=300)
        plt.close(fig)
        
        print(f" -> Saved consistency scatter plot to: {file_name}")

if __name__ == "__main__":
    # Adjust file paths to match your directory structure

    
    create_experiment_boxplots(CSV_DATASET, PLOT_OUTPUT_DIR)
    save_individual_scatter_plots(CSV_DATASET, PLOT_OUTPUT_DIR)