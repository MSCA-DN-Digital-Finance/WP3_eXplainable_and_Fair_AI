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
    if "stock_returns" in df.columns:
        df["stock_returns"] = df["stock_returns"].apply(lambda x: [float(i.strip('%')) / 100 for i in x.split(",")])

    # Convert portfolio values from strings with commas to numeric format
    if pd.api.types.is_object_dtype(df["new_portfolio_value"]):  
        df["new_portfolio_value"] = df["new_portfolio_value"].str.replace(",", "").astype(float)
    if "old_portfolio_value" in df.columns:
        if pd.api.types.is_object_dtype(df["old_portfolio_value"]): 
            df["old_portfolio_value"] = df["old_portfolio_value"].str.replace(",", "").astype(float)

    return df



def plot_dual_line_per_episode(df, metric1, metric2, ylabel, title, filename, legend1, legend2):
    """
    Plots two time series per episode in separate subplots.
    Selects only 5 episodes if more than 5 exist, evenly spaced across the full range.

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
    episodes = sorted(df["episode"].unique())
    num_episodes = len(episodes)

    # Select a subset of episodes if more than 5
    if num_episodes <= 5:
        selected_episodes = episodes
    else:
        mid = num_episodes // 2
        first = 0
        last = num_episodes - 1
        between_first_mid = (first + mid) // 2
        between_mid_last = (mid + last) // 2
        selected_indices = [first, between_first_mid, mid, between_mid_last, last]
        selected_episodes = [episodes[i] for i in selected_indices]

    fig, axes = plt.subplots(1, len(selected_episodes), figsize=(5 * len(selected_episodes), 4), sharey=True)

    if len(selected_episodes) == 1:
        axes = [axes]  # Ensure axes is iterable

    for idx, episode in enumerate(selected_episodes):
        df_episode = df[df["episode"] == episode]
        days = df_episode["day"]

        # Detect if the columns are lists or numeric
        if isinstance(df_episode[metric1].iloc[0], list):
            line1 = df_episode[metric1].apply(lambda x: x[0])  # Adjust this index as needed
            line2 = df_episode[metric2].apply(lambda x: x[1])
        else:
            line1 = df_episode[metric1]
            line2 = df_episode[metric2]

        axes[idx].plot(days, line1, label=legend1, color="#00CC00")
        axes[idx].plot(days, line2, label=legend2, color="#BF40BF")

        axes[idx].set_xlabel("Day")
        axes[idx].set_ylabel(ylabel)
        axes[idx].set_title(f"Ep {episode}")
        axes[idx].legend()
        axes[idx].tick_params(axis='x', rotation=45)

    fig.suptitle(title, fontsize=14)
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
    - reward (str): Name of the reward function (for file naming).
    - filename (str): Path to save the plot.
    """

    # Convert percentage strings to numeric if needed
    if "%" in str(df[metric].iloc[0]):  
        df[metric] = df[metric].str.replace("%", "").astype(float) / 100

    # Get unique episodes
    episodes = sorted(df["episode"].unique())
    num_episodes = len(episodes)

    # Select episodes to plot
    if num_episodes <= 5:
        selected_episodes = episodes
    else:
        mid = num_episodes // 2
        first = 0
        last = num_episodes - 1
        between_first_mid = (first + mid) // 2
        between_mid_last = (mid + last) // 2
        selected_indices = [first, between_first_mid, mid, between_mid_last, last]
        selected_episodes = [episodes[i] for i in selected_indices]

    # Create subplots
    fig, axes = plt.subplots(1, len(selected_episodes), figsize=(5 * len(selected_episodes), 4), sharey=True)

    # Ensure axes is iterable
    if len(selected_episodes) == 1:
        axes = [axes]

    for idx, episode in enumerate(selected_episodes):
        df_episode = df[df["episode"] == episode]
        axes[idx].plot(df_episode["day"], df_episode[metric], label=f"Episode {episode}", color="#B7410E")

        # Set labels and titles
        axes[idx].set_xlabel("Day")
        axes[idx].set_ylabel(ylabel)
        axes[idx].set_title(f"Ep {episode}")
        axes[idx].legend()
        axes[idx].set_xticks(df_episode["day"][::50])
        axes[idx].tick_params(axis='x', rotation=45)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
