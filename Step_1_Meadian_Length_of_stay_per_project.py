## Libraries

# Data manipulation
import pandas as pd
import numpy as np

# Get and save data
import os
from os import path

## Data Path
# Define the path to Load data from
Load_path = 'Clean data/'
Save_path = 'Generated files for ASONAM 2025/Empirical_info/'

## Load data
clean_data_df = pd.read_csv(f'{Load_path}20210620_Data with feature with 60 percent or more missing data removed.csv').drop(['Unnamed: 0'],axis=1)
print(len(clean_data_df)) # Check the number of rows in the cleaned data

# Assuming your DataFrame is named df
# Convert entry and exit dates to datetime
clean_data_df['EntryDate'] = pd.to_datetime(clean_data_df['EntryDate'])
clean_data_df['ExitDate'] = pd.to_datetime(clean_data_df['ExitDate'])

# Calculate length of stay in days
clean_data_df['length_of_stay'] = (clean_data_df['ExitDate'] - clean_data_df['EntryDate']).dt.days
print(clean_data_df['length_of_stay'].head()) # Check the first few values of length_of_stay
# Get max length of stay per project
max_los_per_project = clean_data_df.groupby('ProjectType')['length_of_stay'].max().reset_index()

# # Optional: sort by max length of stay
# max_los_per_project = max_los_per_project.sort_values(by='length_of_stay', ascending=False)

print(max_los_per_project)

# Save the result to a CSV file
max_los_per_project.to_csv(f'{Save_path}max_length_of_stay_per_project.csv', index=False)

# Get mean length of stay per project
mean_los_per_project = clean_data_df.dropna(subset=['length_of_stay']).groupby('ProjectType')['length_of_stay'].mean().reset_index()
print(mean_los_per_project)

# Save the result to a CSV file
mean_los_per_project.to_csv(f'{Save_path}mean_length_of_stay_per_project.csv', index=False)

# Get median length of stay per project
median_los_per_project = clean_data_df.dropna(subset=['length_of_stay']).groupby('ProjectType')['length_of_stay'].median().reset_index()
print(median_los_per_project)

# Save the result to a CSV file
median_los_per_project.to_csv(f'{Save_path}median_length_of_stay_per_project.csv', index=True)

# Get the median of the length of stay of median_los_per_project
median_of_median_los = median_los_per_project['length_of_stay'].median()
print(median_of_median_los) # found 78.0