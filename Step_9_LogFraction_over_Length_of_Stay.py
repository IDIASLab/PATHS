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
Per_length_of_stay = {}

key_names = ['Simulation 1', 'Simulation 2', 'Simulation 3', 'Simulation 4', 'Simulation 5', 'Simulation 6', 'Simulation 7', 'Simulation 8', 'Simulation 9', 'Simulation 10']
Column_names = ['Length_of_trajectory','Simulation 1', 'Simulation 2', 'Simulation 3', 'Simulation 4', 'Simulation 5', 'Simulation 6', 'Simulation 7', 'Simulation 8', 'Simulation 9', 'Simulation 10','Proposed Model', 'GT']

# Initialize the maximum length of trajectory
max_length = 0

# Load the agent history data
for i in range(1,11):
    # if i ==1:
    Agent_history[i] = pd.read_csv(f"{Load_path}agent_histories_5.993k_with_{occurrence_type}_exit_sim_{i}.csv")
    Agent_history[i]['Length_of_stay'] = Agent_history[i]['Timestep_History_String'].apply(
        lambda x: int(x.strip().split(',')[-1]) - int(x.strip().split(',')[0]))
    # print(Agent_history[i].head())
    key = key_names[i-1]
    Per_length_of_stay[key] = Agent_history[i].groupby('Length_of_stay').size().reset_index(name='Frequency')
    # print(Per_trajectory_legth[key].head())
    if(max_length < Agent_history[i]['Length_of_stay'].max()):
        max_length = Agent_history[i]['Length_of_stay'].max()
    # print(Agent_history[i].Length_of_stay.unique())
    # print(key)
print(max_length)

Stay_length_df = pd.DataFrame(
    {
    'Length_of_stay': range(1, max_length + 1)
}
)

for key in key_names:
    # Merge on 'Length_of_trajectory' to align matching values
    Stay_length_df = Stay_length_df.merge(
        Per_length_of_stay[key][['Length_of_stay', 'Frequency']],
        on='Length_of_stay',
        how='left'
    ).rename(columns={'Frequency': key})

print(Stay_length_df.head())

# Calculate the average number of agents across all simulations
Stay_length_df['Avg_simulation'] = Stay_length_df[key_names].mean(axis=1)
print(Stay_length_df.head())

# Load the ground truth data
GT_per_length_of_stay_from2012 = pd.read_csv(f'{Load_path2}Ground_truth_Length_of_stay_{occurrence_type}_exit.csv')


# Merge on 'Length_of_trajectory' to align matching values
Stay_length_df = Stay_length_df.merge(
        GT_per_length_of_stay_from2012[['Length_of_stay', 'Frequency']],
        on='Length_of_stay',
        how='left'
    ).rename(columns={'Frequency': 'GT'})

# Calculate length of stay in years
# Exclude 'Length_of_stay' and work only with value columns
data_only = Stay_length_df.drop(columns='Length_of_stay')

# Group every 12 rows and sum them
Stay_length_df_year = data_only.groupby(
    Stay_length_df.index // 12  # creates group 0 for rows 0–11, group 1 for 12–23, etc.
).sum()

# Add back the new "Length_of_stay" as years 1 to 6
Stay_length_df_year.insert(0, 'Length_of_stay', range(1, len(Stay_length_df_year) + 1))

print(Stay_length_df_year.head())

# Calculate the fraction of agents for each length of stay (months)
# Get the column range to normalize (excluding 'Length_of_stay')
columns_to_normalize = Stay_length_df.columns[1:]  # from 'S1' to 'GT'

# Normalize each column by its column sum
Stay_length_df[columns_to_normalize] = Stay_length_df[columns_to_normalize].div(
    Stay_length_df[columns_to_normalize].sum(axis=0),
    axis=1
)
print(Stay_length_df.head())
# save the DataFrame to a CSV file  
Stay_length_df.to_csv(f'{Load_path}All_Lengths_of_Stay_months_{occurrence_type}_exit.csv', index=False)

# Calculate the fraction of agents for each length of stay (years)
# Get the column range to normalize (excluding 'Length_of_stay')
columns_to_normalize = Stay_length_df_year.columns[1:]  # from 'S1' to 'GT'
# Normalize each column by its column sum
Stay_length_df_year[columns_to_normalize] = Stay_length_df_year[columns_to_normalize].div(
    Stay_length_df_year[columns_to_normalize].sum(axis=0),
    axis=1
)
print(Stay_length_df_year.head())
# save the DataFrame to a CSV file
Stay_length_df_year.to_csv(f'{Load_path}All_Lengths_of_Stay_years_{occurrence_type}_exit.csv', index=False)

def plot_length_of_stay_vs_log_fraction(df, x_col):
    columns_to_plot = df.columns[1:]  # Assuming df is your DataFrame
    x = df['Length_of_stay']

    # Define different markers for each line
    markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', '<', '>', 'h', 'H', '+']
    num_lines = len(columns_to_plot)

    plt.figure(figsize=(12, 6))

    for i, col in enumerate(columns_to_plot):
        marker = markers[i % len(markers)]  # Cycle markers if more columns than marker types

        # Set transparency: full (1.0) for last 2 columns, faded (e.g., 0.3) for others
        alpha = 1.0 if i >= num_lines - 2 else 0.3
        # Plot with transparency
        plt.semilogy(x, df[col], marker=marker, label=col, alpha=alpha)

    # plt.title('Semilog Plot of Trajectory Lengths')
    plt.xlabel(f'Length of Stay (in {x_col})', fontsize=23, labelpad=15)
    plt.xticks(fontsize=23)
    plt.ylabel('Fraction (Log Scale)', fontsize=23, labelpad=15)
    plt.yticks(fontsize=23)
    plt.grid(True)
    plt.legend(title='Duration for', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=18, title_fontsize=19)
    plt.tight_layout()
    plt.savefig(f'{fig_path}All_Length_of_Stay_{x_col}_vs_log_Fraction_{occurrence_type}.png')
    plt.show()

plot_length_of_stay_vs_log_fraction(Stay_length_df,'months')



def plot_log_length_of_stay_vs_log_fraction(df, x_col):
    columns_to_plot = df.columns[1:]  # Assuming df is your DataFrame
    x = df['Length_of_stay']

    # Define different markers for each line
    markers = ['o', 's', '^', 'v', 'D', 'P', '*', 'X', '<', '>', 'h', 'H', '+']
    num_lines = len(columns_to_plot)

    plt.figure(figsize=(12, 6))

    for i, col in enumerate(columns_to_plot):
        marker = markers[i % len(markers)]  # Cycle markers if more columns than marker types

        # Set transparency: full (1.0) for last 2 columns, faded (e.g., 0.3) for others
        alpha = 1.0 if i >= num_lines - 2 else 0.3
        # Plot with transparency
        plt.loglog(x, df[col], marker=marker, label=col, alpha=alpha)

    # plt.title('Semilog Plot of Trajectory Lengths')
    plt.xlabel(f'Log Length of Stay (in {x_col})', fontsize=16, labelpad=15)
    plt.xticks(fontsize=16)
    plt.ylabel('Fraction (Log Scale)', fontsize=16, labelpad=15)
    plt.yticks(fontsize=16)
    plt.grid(True)
    plt.legend(title='Duration for', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, title_fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{fig_path}All_log_Length_of_Stay_{x_col}_vs_log_Fraction_{occurrence_type}.png')
    plt.show()
