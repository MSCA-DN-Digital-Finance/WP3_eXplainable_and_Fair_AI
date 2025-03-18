import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def preprocess_df(df):
    """
    Preprocess the dataframe by converting relevant columns into lists or numeric values.
    
    - Converts "actions", "allocation_weights", and "stock_returns" into lists.
    - Converts "new_portfolio_value" and "old_portfolio_value" into numeric values.
    
    Parameters:
    - df (pd.DataFrame): The raw dataframe containing training logs.
    
    Returns:
    - df (pd.DataFrame): The preprocessed dataframe.
    """
    df = df.copy()  # Avoid modifying original dataframe

    # Convert string representations of lists into actual lists
    df["actions"] = df["actions"].apply(lambda x: [float(i) for i in x.strip("[]").split(",")])
    df["allocation_weights"] = df["allocation_weights"].apply(lambda x: [float(i) for i in x.strip("[]").split(",")])
    df["stock_returns"] = df["stock_returns"].apply(lambda x: [float(i.strip('%')) / 100 for i in x.split(",")])

    # Convert portfolio values from strings with commas to numeric format
    if pd.api.types.is_object_dtype(df["new_portfolio_value"]):  
        df["new_portfolio_value"] = df["new_portfolio_value"].str.replace(",", "").astype(float)
    if pd.api.types.is_object_dtype(df["old_portfolio_value"]): 
        df["old_portfolio_value"] = df["old_portfolio_value"].str.replace(",", "").astype(float)

    return df

def plot_dual_line_per_episode(df, metric1, metric2, ylabel, title, filename, legend1, legend2):
    """
    Plots two time series per episode in separate subplots.

    Handles cases where `metric1` and `metric2` are:
    - Lists (e.g., actions, allocation weights, stock returns).
    - Numeric values (e.g., old and new portfolio values).

    Parameters:
    - df (pd.DataFrame): Dataframe containing the data.
    - metric1 (str): First metric to plot.
    - metric2 (str): Second metric to plot.
    - ylabel (str): Label for the Y-axis.
    - title (str): Title for the overall figure.
    - filename (str): Path to save the plot.
    - legend1 (str): Legend label for the first metric.
    - legend2 (str): Legend label for the second metric.
    """
    episodes = df["episode"].unique()
    num_episodes = len(episodes)

    fig, axes = plt.subplots(1, num_episodes, figsize=(5 * num_episodes, 4), sharey=True)

    if num_episodes == 1:
        axes = [axes]  # Ensure axes is iterable for a single episode

    for idx, episode in enumerate(episodes):
        df_episode = df[df["episode"] == episode]
        days = df_episode["day"]

        # Detect if the columns are lists or numeric
        if isinstance(df_episode[metric1].iloc[0], list):
            line1 = df_episode[metric1].apply(lambda x: x[0])  # Extract first component if list
            line2 = df_episode[metric2].apply(lambda x: x[1])
        else:
            line1 = df_episode[metric1]  # Directly use the column if numeric
            line2 = df_episode[metric2]

        axes[idx].plot(days, line1, label=legend1, color="#00CC00")  # Green
        axes[idx].plot(days, line2, label=legend2, color="#BF40BF")  # Purple

        axes[idx].set_xlabel("Day")
        axes[idx].set_ylabel(ylabel)  # Set Y-axis label 
        axes[idx].set_title(f"Ep {episode}")
        axes[idx].legend()

    fig.suptitle(title, fontsize=14)  # Global title
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()


def plot_per_episode_row(df, metric, ylabel, title, filename):
    """
    Plot multiple time series in a single row with separate subplots per episode.

    Parameters:
    - df (pd.DataFrame): The dataset containing the episodes.
    - metric (str): Column name of the metric to plot.
    - ylabel (str): Y-axis label.
    - title (str): Plot title.
    - filename (str): Path to save the plot.
    - reward (str): Name of the reward function (for file naming).
    - yticks (int): Number of y-ticks.
    """
    # Convert percentage strings to numeric if needed
    if "%" in str(df[metric].iloc[0]):  
        df[metric] = df[metric].str.replace("%", "").astype(float) / 100

    # Get unique episodes
    episodes = df["episode"].unique()
    num_episodes = len(episodes)

    # Create subplots
    fig, axes = plt.subplots(1, num_episodes, figsize=(5 * num_episodes, 4), sharey=True)

    # Ensure axes is iterable for a single episode
    if num_episodes == 1:
        axes = [axes]

    for idx, episode in enumerate(episodes):
        df_episode = df[df["episode"] == episode]
        axes[idx].plot(df_episode["day"], df_episode[metric], label=f"Episode {episode}", color="#B7410E")

        # Set labels and titles
        axes[idx].set_xlabel("Day")
        axes[idx].set_ylabel(ylabel)  # Set Y-axis label 
        axes[idx].set_title(f"Ep {episode}")
        axes[idx].legend()

        # Reduce the number of x-axis ticks
        axes[idx].set_xticks(df_episode["day"][::50])  
        axes[idx].tick_params(axis='x', rotation=45)  


    fig.suptitle(title, fontsize=14)  
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()