## Libraries

# Data manipulation
import pandas as pd
import numpy as np
from scipy.stats import entropy
from scipy.special import kl_div

# Get and save data
import os
from os import path

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import IPython
from mpl_toolkits.mplot3d import Axes3D


## Data Path
# Define the path to Load data from
Load_path = 'Clean data/'
Save_path = 'Generated files for ASONAM 2025/Emperical_info/'
fig_path = 'Final_Results/'


# Change this to 
# 'median_los_imputation_for_exit_data_last_occurrence', 
# 'median_los_imputation_for_exit_data_last_four_occurrence', 
# as needed
occurrence_type = 'median_los_imputation_for_exit_data_last_occurrence' 

# ## Load data
ground_truth_Median_los_imputation_for_exit_data_df = pd.read_csv(f'{Save_path}Entry_per_timestep_per_project_from_2012_using_{occurrence_type}_with_entry.csv')

# simulation_Median_los_last_four_occurrence_df = pd.read_csv(f'{Save_path}TransitionModel_results_5.993k_with_median_los_imputation_for_exit_data_last_four_occurrence_exit.csv')

# Project Types in the dataset
project_types_info = [0,1,2,3,4,6,11,12,13,14,-3]
project_types_names = ['Entry','P1','P2','P3','P4','P6','P11','P12','P13','P14','Exit']

# Total number of homeless people
Number_of_homeless_people = 5993

# Prepare the Ground truth data
ground_truth = ground_truth_Median_los_imputation_for_exit_data_df.copy()
ground_truth = ground_truth[ground_truth['Timestep']<73] # Filter out the first 72 timesteps

# Add column for ProjectType names in the ground truth data
ground_truth['ProjectTypeName'] = ground_truth_Median_los_imputation_for_exit_data_df['ProjectType'].map(dict(zip(project_types_info, project_types_names)))
print(ground_truth.head())

# Increase the timesteps by 1
ground_truth['Timestep'] = ground_truth_Median_los_imputation_for_exit_data_df['Timestep'] + 1
print(ground_truth.head())

#make adjacency matrix using timestep as index and project type as columns
ground_truth_pivot = ground_truth.pivot(index='Timestep', columns='ProjectTypeName', values='Number_of_Entry_per_Month').fillna(0)
print(ground_truth_pivot.head())

# rearrange the column in order ['Entry', 'P1','P2','P3','P4','P6','P11','P12','P13','P14','Exit']
ground_truth_pivot = ground_truth_pivot[['Entry', 'P1', 'P2', 'P3', 'P4', 'P6', 'P11', 'P12', 'P13', 'P14', 'Exit']]

# ground_truth_pivot = ground_truth_pivot[[col for col in ground_truth_pivot.columns if col != 'Exit'] + ['Exit']]
print(ground_truth_pivot.head())
print(ground_truth_pivot.shape)

# Save the ground truth data
ground_truth_pivot.to_csv(f'{Save_path}Ground_truth_{occurrence_type}_over_timestep.csv')
