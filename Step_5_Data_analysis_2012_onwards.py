## Figure out the entries per timestep

## Libraries

# Data manipulation
import pandas as pd
import numpy as np

# Get and save data
import os
from os import path

## Data Path
# Define the path to Load data from
File_path = 'Generated files for ASONAM 2025/Empirical_info/'

# Change this to 
# 'median_los_imputation_for_exit_data_last_occurrence', 
# 'median_los_imputation_for_exit_data_last_four_occurrence', 
# as needed
occurrence_type = 'median_los_imputation_for_exit_data_last_four_occurrence' 

## Load data
exit_last_occurrence_from2012 = pd.read_csv(f'{File_path}Clean_data_using_missing_exit_filled_with_median_los_2012_onwards_-3_at_last_occurrence.csv')
exit_last_four_occurrence_from2012 = pd.read_csv(f'{File_path}Clean_data_using_missing_exit_filled_with_median_los_2012_onwards_last_four_occurrence.csv')


if occurrence_type == 'median_los_imputation_for_exit_data_last_occurrence':
    clean_data_df = exit_last_occurrence_from2012.copy() 
elif occurrence_type == 'median_los_imputation_for_exit_data_last_four_occurrence':
    clean_data_df = exit_last_four_occurrence_from2012.copy()
print(clean_data_df.head())


## Capacity of project types
# Convert column to datetime type
clean_data_df['EntryDate'] = pd.to_datetime(clean_data_df['EntryDate'])
clean_data_df['ExitDate'] = pd.to_datetime(clean_data_df['ExitDate'])

# Extract only the year
clean_data_df['EntryYear'] = clean_data_df['EntryDate'].dt.year
# clean_data_df['ExitYear'] = clean_data_df['ExitDate'].dt.year

clean_data_df_from2012= clean_data_df[(clean_data_df['EntryYear'] >= 2012)]

clean_data_df_from2012['EntryMonth'] = clean_data_df_from2012['EntryDate'].dt.month
# clean_data_df_from2012['ExitMonth'] = clean_data_df_from2012['ExitDate'].dt.month

clean_data_df_from2012.to_csv(f'{File_path}Clean_data_from2012.csv', index=False) 

# print(clean_data_df_from2012[clean_data_df_from2012['ExitYear'] >= 2012].EntryYear)
print(clean_data_df_from2012.groupby(['ClientID', 'ProjectType'])[['ClientID', 'ProjectType']]
      .apply(lambda x: x.sort_values(by=['ClientID'])))

# Keep only relevant columns
clean_data_df_from2012 = clean_data_df_from2012[["ClientID", "EntryYear", "EntryMonth", "ProjectType"]]
# Concatenate the clean data with exit data
Data_df_from_2012 = clean_data_df_from2012.copy()
# Data_df_from_2012 = pd.concat([clean_data_df_from2012, exit_data_df_from2012], axis=0, ignore_index=True)

##population of clean data
population = len(clean_data_df_from2012['ClientID'].unique())
print(population) # Found 5993

# ## population of Data_df_from_2012
# Total_population = len(Data_df_from_2012['ClientID'].unique())
# print(Total_population) # Found 34247


## Entry in each month of each year
Entry_per_project_per_month_from2012 = Data_df_from_2012.groupby(['EntryYear','EntryMonth','ProjectType'])['ProjectType'].value_counts().reset_index(name='Number_of_Entry_per_Month')
print(Entry_per_project_per_month_from2012.head())
Entry_per_project_per_month_from2012.to_csv(f'{File_path}Entry_per_project_per_month_from_2012_using_{occurrence_type}.csv', index=False) 

timestep_counter = 1
# Initialize the Timestep column with NaN
Entry_per_project_per_month_from2012['Timestep'] = np.nan
for year in Entry_per_project_per_month_from2012['EntryYear'].unique():
    for month in Entry_per_project_per_month_from2012['EntryMonth'].unique():
        # Get the subset for the current year and month
        subset = Entry_per_project_per_month_from2012[(Entry_per_project_per_month_from2012['EntryYear'] == year) & (Entry_per_project_per_month_from2012['EntryMonth'] == month)]
        Entry_per_project_per_month_from2012.loc[subset.index, 'Timestep'] = timestep_counter
        timestep_counter += 1
print(Entry_per_project_per_month_from2012)

# Save the final DataFrame with Timestep to a CSV file
Entry_per_project_per_month_from2012[['Timestep', 'ProjectType','Number_of_Entry_per_Month']].to_csv(f'{File_path}Entry_per_timestep_per_project_from_2012_using_{occurrence_type}.csv', index=False)
print(Entry_per_project_per_month_from2012[['Timestep', 'ProjectType','Number_of_Entry_per_Month']].head())