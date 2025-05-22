## Libraries

# Data manipulation
import pandas as pd
import numpy as np

# Get and save data
import os
from os import path

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
Save_path = 'Generated files for ASONAM 2025/Empirical_info/'
fig_path = 'Final_Results/'

# Change this to 
# 'median_los_imputation_for_exit_data_last_occurrence',
# 'median_los_imputation_for_exit_data_last_four_occurrence'
occurrence_type = 'median_los_imputation_for_exit_data_last_occurrence' 

## Load Data
if occurrence_type == 'median_los_imputation_for_exit_data_last_occurrence':
    file_name = 'Clean_data_using_missing_exit_filled_with_median_los_2012_onwards_-3_at_last_occurrence.csv'
elif occurrence_type == 'median_los_imputation_for_exit_data_last_four_occurrence':
    file_name = 'Clean_data_using_missing_exit_filled_with_median_los_2012_onwards_last_four_occurrence.csv'
clean_data_df = pd.read_csv(f'{Save_path}{file_name}')

##population of clean data
population = len(clean_data_df['ClientID'].unique())
population # Found 6011

# Convert column to datetime type
clean_data_df['EntryDate'] = pd.to_datetime(clean_data_df['EntryDate'])
clean_data_df['ExitDate'] = pd.to_datetime(clean_data_df['ExitDate'])

# Take the data upto 2017
clean_data_df = clean_data_df[clean_data_df['EntryDate'] < '2018-01-01']

# Extract only the year
clean_data_df['EntryYear'] = clean_data_df['EntryDate'].dt.year
clean_data_df['ExitYear'] = clean_data_df['ExitDate'].dt.year

clean_data_df['EntryMonth'] = clean_data_df['EntryDate'].dt.month
clean_data_df['ExitMonth'] = clean_data_df['ExitDate'].dt.month

print(clean_data_df.head())

## Initial entry per project
initial_entry = clean_data_df.drop_duplicates(subset='ClientID', keep='first')
initial_entry = initial_entry[['ClientID', 'ProjectType']]
print(initial_entry)

## Drop rows where ProjectType = -3
initial_entry = initial_entry[initial_entry['ProjectType'] != -3]

## Calculate weights for the initial entry per project
count_initial_entry_per_project = initial_entry.groupby('ProjectType').count().reset_index()
count_initial_entry_per_project.columns = ['ProjectType', 'Count']
count_initial_entry_per_project["Weight"] = (count_initial_entry_per_project["Count"]/count_initial_entry_per_project["Count"].sum())
print(count_initial_entry_per_project)

count_initial_entry_per_project.to_csv(f'{Save_path}initial_entry_weight_from_cleaned_data_2012_upto_2017_{occurrence_type}.csv', index=False) 
print(sum(count_initial_entry_per_project["Weight"])) # Found 1.0

# ## Entry per project per month
First_occurrences_per_individual = clean_data_df.drop_duplicates(subset=['ClientID'], keep='first')
First_occurrence_per_month = First_occurrences_per_individual.groupby(['EntryYear', 'EntryMonth']).size().reset_index(name='Number_of_Entry_per_Month')
print(First_occurrence_per_month.head())

# Save this data to a CSV file
if occurrence_type == 'median_los_imputation_for_exit_data_last_occurrence':
    First_occurrence_per_month.to_csv(f'{Save_path}Entry_per_timestep_using_first_occurrence.csv', index=False)
elif occurrence_type == 'median_los_imputation_for_exit_data_last_four_occurrence':
    First_occurrence_per_month.to_csv(f'{Save_path}Entry_per_timestep_using_last_four_occurrence.csv', index=False)

# Data from 2012 onwards
clean_data_df_2012_onwards = clean_data_df[(clean_data_df['EntryYear'] >= 2012)]
First_occurrences_per_individual_2012_onwards = clean_data_df_2012_onwards.drop_duplicates(subset=['ClientID'], keep='first')
First_occurrence_per_month_2012_onwards = First_occurrences_per_individual_2012_onwards.groupby(['EntryYear', 'EntryMonth']).size().reset_index(name='Number_of_Entry_per_Month')
First_occurrence_per_month_2012_onwards = First_occurrence_per_month_2012_onwards.reset_index().rename(columns={"index": "Timestep"})
First_occurrence_per_month_2012_onwards['ProjectType'] = 0
print(First_occurrence_per_month_2012_onwards.head())
# Save this data to a CSV file
if occurrence_type == 'median_los_imputation_for_exit_data_last_occurrence':
    First_occurrence_per_month_2012_onwards.to_csv(f'{Save_path}Entry_per_timestep_from_2012_using_first_occurrence.csv', index=False)
elif occurrence_type == 'median_los_imputation_for_exit_data_last_four_occurrence':
    First_occurrence_per_month_2012_onwards.to_csv(f'{Save_path}Entry_per_timestep_from_2012_using_last_four_occurrence.csv', index=False)

