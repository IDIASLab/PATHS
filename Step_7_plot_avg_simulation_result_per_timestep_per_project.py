## Libraries
# Model design
import agentpy as ap
import networkx as nx
import random

# Visualization
import matplotlib.pyplot as plt
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
File_path = 'Generated files for ASONAM 2025/Multiple_simulations/Using_all_2012_2017/'
fig_path = 'Final_Results//Using_all_2012_2017/'

# Output data from the simulation
# Change this to 
# 'median_los_imputation_for_exit_data_last_occurrence', 
# 'median_los_imputation_for_exit_data_last_four_occurrence', 
# as needed
occurrence_type = 'median_los_imputation_for_exit_data_last_occurrence' 

if occurrence_type == 'median_los_imputation_for_exit_data_last_four_occurrence':
    File_path = 'Generated files for ASONAM 2025/Multiple_simulations/Using_last_four_occurrences/'
    fig_path = 'Final_Results/Using_last_four_occurrences/'


# Total number of homeless people
Number_of_homeless_people = 5993

sim = {}
## Load data
for i in range(1,11):
    sim[i] = pd.read_csv(f'{File_path}TransitionModel_results_{(Number_of_homeless_people/1000)}k_with_{occurrence_type}_exit_sim_{i}.csv')

print(f"Loaded {len(sim)} simulation files.")

print(sim[1].Entry.head())

# Average the results across all simulations
avg_results = sim[1].copy() 
for i in range(2, 11):
    avg_results += sim[i]
    print(f' {i} : \n {avg_results.head()}')

avg_results /= 10

print(avg_results.head())
#Drop the first row
avg_results = avg_results.drop(index=0)
print(avg_results.head())

# Save the averaged results
avg_results.to_csv(f'{File_path}Average_TransitionModel_results_{(Number_of_homeless_people/1000)}k_with_{occurrence_type}_exit.csv', index=False)

# Define the fixed project types and names
project_types_info = [0,1,2,3,4,6,11,12,13,14,-3]
project_types_names = ['Entry','P1','P2','P3','P4','P6','P11','P12','P13','P14','Exit']



def homeless_3dplot(results):
    """ 3D plot of people's condition over time. """
    
    # Convert recorded results to a DataFrame
    df = pd.DataFrame(results)

    # Reset index to include the timestep as a column
    df = df.reset_index()
    
    # Set up 3D plot
    fig = plt.figure(constrained_layout=True, figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create a color palette
    colors = sns.color_palette("tab10", len(project_types_names))

    # Skip the first row (timestep 0)
    df = df.iloc[1:]

    # Plot each project type over timesteps
    for i, project in enumerate(project_types_names):
        # if i !=0:
        ax.plot(df["t"], [i] * len(df), df[project], color=colors[i], label=project)

    # Axis labels
    ax.set_xlabel("Timestep (in months)", fontsize=23, labelpad=23)
    ax.set_ylabel("Services", fontsize=22, labelpad=23)
    ax.set_zlabel("Population Frequency", fontsize=23, labelpad=23)

    # Set y-axis to project type names
    ax.set_yticks(range(len(project_types_info)))
    ax.set_yticklabels(project_types_names, fontsize=22, rotation=-20)
    ax.yaxis.set_tick_params(pad=5)  # Add padding to y-axis ticks
    ax.tick_params(axis='x', labelsize=22) 
    ax.tick_params(axis='z', labelsize=22)
    ax.zaxis.set_tick_params(pad=10)  

    # Show legend
    ax.legend(
        title='Services',
        title_fontsize=23,
        loc="upper left",
        bbox_to_anchor=(1.1, 1),
        fontsize=22
        )
    ax.grid(True)

    plt.show()
    

homeless_3dplot(avg_results)
