## Libraries

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import IPython
from mpl_toolkits.mplot3d import Axes3D

# Data manipulation
import pandas as pd
import numpy as np


# Get and save data
import os
from os import path

## Data Path
# Define the path to Load data from
Load_path = 'Generated files for ASONAM 2025/Emperical_info/'
fig_path = 'Final_Results/'

## Load data
Per_timestep_transition_median_los_imputation_for_exit_data_last_occurrence_df = pd.read_csv(f'{Load_path}Entry_per_timestep_per_project_from_2012_using_median_los_imputation_for_exit_data_last_occurrence.csv')
Per_timestep_transition_median_los_imputation_for_exit_data_last_four_occurrence_df = pd.read_csv(f'{Load_path}Entry_per_timestep_per_project_from_2012_using_median_los_imputation_for_exit_data_last_four_occurrence.csv')

Per_timestep_new_entry = pd.read_csv(f'{Load_path}Entry_per_timestep_from_2012_using_first_occurrence.csv')
Per_timestep_new_entry = Per_timestep_new_entry[["Timestep", "ProjectType", "Number_of_Entry_per_Month"]]
print(Per_timestep_new_entry.head())

# Change this to 
# 'median_los_imputation_for_exit_data_last_occurrence', 
# 'median_los_imputation_for_exit_data_last_four_occurrence', 
#  as needed
occurrence_type = 'median_los_imputation_for_exit_data_last_occurrence'

if occurrence_type == 'median_los_imputation_for_exit_data_last_four_occurrence':
    Per_timestep_transition_df = pd.concat([Per_timestep_transition_median_los_imputation_for_exit_data_last_four_occurrence_df, Per_timestep_new_entry], ignore_index=True) 
elif occurrence_type == 'median_los_imputation_for_exit_data_last_occurrence':
    Per_timestep_transition_df = pd.concat([Per_timestep_transition_median_los_imputation_for_exit_data_last_occurrence_df, Per_timestep_new_entry], ignore_index=True) 
Per_timestep_transition_df = Per_timestep_transition_df.sort_values(by=['Timestep', 'ProjectType'])
Per_timestep_transition_df = Per_timestep_transition_df[Per_timestep_transition_df['Timestep'] < 73]
print(Per_timestep_transition_df.head(10))

 

# Total number of homeless people
Number_of_homeless_people = 5993
# Per_timestep_transition_df['Number_of_Entry_per_Month'] = Per_timestep_transition_df['Number_of_Entry_per_Month'] *10 / Number_of_homeless_people 

def ground_truth_homeless_3dplot(ground_truth_df, fig_path, Number_of_homeless_people, occurrence_type):
    """ 3D plot of people's condition over time. """

    # Define the fixed project types and names
    project_types_info = [0,1,2,3,4,6,11,12,13,14,-3]
    project_types_names = ['Entry','P1','P2','P3','P4','P6','P11','P12','P13','P14','Exit']

    # Pivot the data to get ProjectType as columns, Timestep as index
    df_pivot = ground_truth_df.pivot_table(
        index='Timestep',
        columns='ProjectType',
        values='Number_of_Entry_per_Month',
        fill_value=0
    ).reset_index()

    # Set up the 3D plot
    fig = plt.figure(constrained_layout=True, figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create a color palette
    colors = sns.color_palette("tab10", len(project_types_info))

    # Plot each project type line
    for i, pt in enumerate(project_types_info):
        if pt in df_pivot.columns:
            ax.plot(
                df_pivot["Timestep"],             # x-axis: Timestep
                [i] * len(df_pivot),              # y-axis: project index (mapped to label)
                df_pivot[pt],                     # z-axis: number of entries
                color=colors[i],
                label=project_types_names[i]
            )

    # Set axis labels
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
ground_truth_homeless_3dplot(Per_timestep_transition_df, fig_path, Number_of_homeless_people, occurrence_type)

