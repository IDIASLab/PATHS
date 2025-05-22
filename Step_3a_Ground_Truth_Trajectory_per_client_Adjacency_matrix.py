import pandas as pd
import numpy as np
from scipy.stats import entropy
from scipy.special import kl_div

# Get and save data
import os
from os import path

# Visualization
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter
import seaborn as sns
import IPython
from mpl_toolkits.mplot3d import Axes3D


## Data Path
# Define the path to Load data from
Load_path = 'Clean data/'
Save_path = 'Generated files for ASONAM 2025/'
fig_path = 'Final_Results/'

# Change this to 
# 'median_los_imputation_for_exit_data_last_occurrence',
# 'median_los_imputation_for_exit_data_last_four_occurrence'
occurrence_type = 'median_los_imputation_for_exit_data_last_four_occurrence' 

## Load Data
if occurrence_type == 'median_los_imputation_for_exit_data_last_occurrence':
    file_name = 'Clean_data_using_missing_exit_filled_with_median_los_2012_onwards_-3_at_last_occurrence.csv'
elif occurrence_type == 'median_los_imputation_for_exit_data_last_four_occurrence':
    file_name = 'Clean_data_using_missing_exit_filled_with_median_los_2012_onwards_last_four_occurrence.csv'
clean_data_2012_onwards = pd.read_csv(f'{Save_path}{file_name}')

## datetime to pandas datetime
clean_data_2012_onwards['EntryDate'] = pd.to_datetime(clean_data_2012_onwards['EntryDate'])
clean_data_2012_onwards['ExitDate'] = pd.to_datetime(clean_data_2012_onwards['ExitDate'])

## Take the data upto 2017
clean_data_2012_onwards = clean_data_2012_onwards[clean_data_2012_onwards['EntryDate'] < '2018-01-01']


## Calculate the trajectories for each of the ClientID
Trajectory_df = pd. DataFrame()
Trajectory_df['ClientID'] = clean_data_2012_onwards['ClientID'].unique()

for i in range(1,68):
    Trajectory_df[str(i)] = None

for i in range(len(Trajectory_df)):
    ClientID = Trajectory_df['ClientID'][i]
    trajectory = clean_data_2012_onwards[clean_data_2012_onwards['ClientID'] == ClientID]
    trajectory = trajectory[['ClientID', 'EntryDate','ProjectType' ]]
    trajectory = trajectory.reset_index(drop=True)
    # append each ProjectType in a row of the dataframe
    for j in range(len(trajectory)):
        Trajectory_df[str(j+1)][i] = trajectory['ProjectType'][j]      
print(Trajectory_df.head())

# Save the trajectory data
Trajectory_df.to_csv(f'{Save_path}GT_Trajectory_df_{occurrence_type}.csv', index=False)

# Define the custom labels for both rows and columns
labels = [1, 2, 3, 4, 6, 11, 12, 13, 14, -3]

# Create the DataFrame with all values initialized to 0 (or any other default)
Adjacency_matrix = pd.DataFrame(0, index=labels, columns=labels)

print(Adjacency_matrix)

# Count the occurrences of each pair of ProjectType
for i in range(len(Trajectory_df)):
    for j in range (1,len(Trajectory_df.columns)-1):
        source = Trajectory_df[str(j)][i]
        target = Trajectory_df[str(j+1)][i]
        if source != None and target != None:
            Adjacency_matrix[target][source] += 1 

print(Adjacency_matrix)

# ROW WISE NORMALIZATION
Adjacency_matrix = Adjacency_matrix.div(Adjacency_matrix.sum(axis=1), axis=0)
print(Adjacency_matrix)

# Rename the index to 'Source/Target'
Adjacency_matrix.index.name = 'Source/Target'

# Save the Adjacency matrix
Adjacency_matrix.to_csv(f'{Save_path}Empirical_Adjacency_matrix_{occurrence_type}.csv', index=True)
