## Libraries
# Model design
import agentpy as ap
import networkx as nx
import random

# Visualization
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter
import seaborn as sns
import IPython
from mpl_toolkits.mplot3d import Axes3D

# Data manipulation
import pandas as pd
import numpy as np
from scipy.stats import entropy

# Get and save data
import os
from os import path

## Data Path
# Define the path to Load data from
Load_path = 'Generated files for ASONAM 2025/Multiple_simulations/Using_all_2012_2017/'
Load_path2 = 'Generated files for ASONAM 2025/Empirical_info/'
fig_path = 'Final_Results/Using_all_2012_2017/'

# Change this to 
# 'median_los_imputation_for_exit_data_last_occurrence', 
# 'median_los_imputation_for_exit_data_last_four_occurrence', 
# as needed
occurrence_type = 'median_los_imputation_for_exit_data_last_occurrence' 

if occurrence_type == 'median_los_imputation_for_exit_data_last_four_occurrence':
    Load_path = 'Generated files for ASONAM 2025/Multiple_simulations/Using_last_four_occurrences/'
    fig_path = 'Final_Results/Using_last_four_occurrences/'
# Total number of homeless people
Number_of_homeless_people = 5993

# Define dictonaries to store data
Agent_history = {}
Per_trajectory_legth = {}

key_names = ['Simulation 1', 'Simulation 2', 'Simulation 3', 'Simulation 4', 'Simulation 5', 'Simulation 6', 'Simulation 7', 'Simulation 8', 'Simulation 9', 'Simulation 10']
Column_names = ['Length_of_trajectory','Simulation 1', 'Simulation 2', 'Simulation 3', 'Simulation 4', 'Simulation 5', 'Simulation 6', 'Simulation 7', 'Simulation 8', 'Simulation 9', 'Simulation 10','Proposed Model', 'GT']

# Initialize the maximum length of trajectory
max_length = 0

# Load the agent history data
for i in range(1,11):
    # if i ==1:
    Agent_history[i] = pd.read_csv(f"{Load_path}agent_histories_5.993k_with_{occurrence_type}_exit_sim_{i}.csv")
    Agent_history[i]['Length_of_trajectory'] = Agent_history[i]['Project_History_String'].str.split(',').str.len()
    # print(Agent_history[i].head())
    key = key_names[i-1]
    Per_trajectory_legth[key] = Agent_history[i].groupby('Length_of_trajectory').size().reset_index(name='Number_of_agents')
    # print(Per_trajectory_legth[key].head())
    if(max_length < Agent_history[i]['Length_of_trajectory'].max()):
        max_length = Agent_history[i]['Length_of_trajectory'].max()
    # print(Agent_history[i].Length_of_trajectory.unique())
    # print(key)

Trajectory_length_df = pd.DataFrame(
    {
    'Length_of_trajectory': range(1, max_length + 1)
}
)

for key in key_names:
    # Merge on 'Length_of_trajectory' to align matching values
    Trajectory_length_df = Trajectory_length_df.merge(
        Per_trajectory_legth[key][['Length_of_trajectory', 'Number_of_agents']],
        on='Length_of_trajectory',
        # how='left'
    ).rename(columns={'Number_of_agents': key})

# print(Trajectory_length_df.head())

# Calculate the average number of agents across all simulations
Trajectory_length_df['Proposed Model'] = Trajectory_length_df[key_names].mean(axis=1)
# print(Trajectory_length_df.head())

# Load the ground truth data
ground_truth_from2012 = pd.read_csv(f'{Load_path2}GT_Trajectory_df_{occurrence_type}.csv')
ground_truth_from2012 = ground_truth_from2012.replace(to_replace=[None], value=np.nan)
row_lengths = ground_truth_from2012.notna().sum(axis=1)
print(row_lengths)
# 
# Agents per trajectory length
GT_per_trajectory_length = row_lengths.value_counts().sort_index().reset_index(name='Length_of_trajectory')
GT_per_trajectory_length.columns = ['Length_of_trajectory', 'Number_of_agents']
print(GT_per_trajectory_length.head())

# get statistics
# Weighted mean
mean = np.average(GT_per_trajectory_length['Length_of_trajectory'], weights=GT_per_trajectory_length['Number_of_agents'])

# Weighted median
def weighted_median(data, weights):
    sorted_data, sorted_weights = zip(*sorted(zip(data, weights)))
    cum_weights = np.cumsum(sorted_weights)
    cutoff = cum_weights[-1] / 2.0
    return sorted_data[np.where(cum_weights >= cutoff)[0][0]]

median = weighted_median(GT_per_trajectory_length['Length_of_trajectory'], GT_per_trajectory_length['Number_of_agents'])

print(f"Weighted Mean: {mean}")
print(f"Weighted Median: {median}")

# Merge on 'Length_of_trajectory' to align matching values
Trajectory_length_df = Trajectory_length_df.merge(
        GT_per_trajectory_length[['Length_of_trajectory', 'Number_of_agents']],
        on='Length_of_trajectory',
        how='right'
    ).rename(columns={'Number_of_agents': 'GT'})

# Nan to 0
# Trajectory_length_df = Trajectory_length_df.fillna(0)

# Calculate the fraction of agents for each trajectory length
# Get the column range to normalize (excluding 'Length_of_trajectory')
columns_to_normalize = Trajectory_length_df.columns[1:]  # from 'S1' to 'GT'

# Normalize each column by its column sum
Trajectory_length_df[columns_to_normalize] = Trajectory_length_df[columns_to_normalize].div(
    Trajectory_length_df[columns_to_normalize].sum(axis=0),
    axis=1
)
print(Trajectory_length_df.head())
# save the DataFrame to a CSV file  
Trajectory_length_df.to_csv(f'{Load_path}All_Lengths_of_Trajectory_{occurrence_type}_exit.csv', index=False)

def plot_length_of_trajectory_vs_log_fraction(df):
    columns_to_plot = df.columns[1:]  # Assuming df is your DataFrame
    x = df['Length_of_trajectory']

    # Define different markers for each line
    markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', '<', '>', 'h', 'H', '+']
    num_lines = len(columns_to_plot)

    plt.figure(figsize=(12, 6))

    for i, col in enumerate(columns_to_plot):
        marker = markers[i % len(markers)]  # Cycle markers if more columns than marker types

        # Set transparency: full (1.0) for last 2 columns, faded (e.g., 0.3) for others
        alpha = 1.0 if i >= num_lines - 2 else 0.3

        plt.semilogy(x, df[col], marker=marker, label=col, alpha=alpha)

    # plt.title('Semilog Plot of Trajectory Lengths')
    plt.xlabel('Length of Trajectory', fontsize=16, labelpad=15)
    plt.xticks(fontsize=16)
    plt.ylabel('Fraction (Log Scale)', fontsize=16, labelpad=15)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.legend(title='Path Length for', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, title_fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{fig_path}All_Length_of_Trajectory_vs_log_Fraction_{occurrence_type}.png')
    plt.show()

# plot_length_of_trajectory_vs_log_fraction(Trajectory_length_df[Trajectory_length_df['Length_of_trajectory'] != 1])

def plot_log_length_of_trajectory_vs_log_fraction(df):
    columns_to_plot = df.columns[1:]  # Assuming df is your DataFrame
    x = df['Length_of_trajectory']

    # Define different markers for each line
    markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', '<', '>', 'h', 'H', '+']
    num_lines = len(columns_to_plot)

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, col in enumerate(columns_to_plot):
        marker = markers[i % len(markers)]  # Cycle markers if more columns than marker types

        # Set transparency: full (1.0) for last 2 columns, faded (e.g., 0.3) for others
        alpha = 1.0 if i >= num_lines - 2 else 0.3

        plt.loglog(x, df[col], marker=marker, label=col, alpha=alpha)

    # plt.title('Semilog Plot of Trajectory Lengths')
    plt.xlabel('Length of Trajectory (Log Scale)', fontsize=23, labelpad=15)
    plt.xticks(fontsize=23)
    plt.ylabel('Fraction (Log Scale)', fontsize=23, labelpad=15)
    plt.yticks(fontsize=23)
    # plt.tick_params(axis='y', labelsize=22, pad=10)
    plt.grid(True)
    plt.legend(title='Path Length for', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=17, title_fontsize=18)
    plt.tight_layout()
    plt.savefig(f'{fig_path}All_log_Length_of_Trajectory_vs_log_Fraction_{occurrence_type}.png')
    plt.show()   

plot_log_length_of_trajectory_vs_log_fraction(Trajectory_length_df[Trajectory_length_df['Length_of_trajectory'] != 1])