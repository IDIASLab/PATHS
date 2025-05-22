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
print(clean_data_df.head())

##population of clean data
population = len(clean_data_df['ClientID'].unique())
print(population) # Found 6011

## Capacity of project types
# Convert column to datetime type
clean_data_df['EntryDate'] = pd.to_datetime(clean_data_df['EntryDate'])
clean_data_df['ExitDate'] = pd.to_datetime(clean_data_df['ExitDate'])

# Extract only the year
clean_data_df['EntryYear'] = clean_data_df['EntryDate'].dt.year
clean_data_df['ExitYear'] = clean_data_df['ExitDate'].dt.year

print(clean_data_df.head())

# Group by ProjectType and EntryYear and count the number of entries
Entry_per_project_per_year = clean_data_df.groupby(['ProjectType','EntryYear'])['ProjectType'].value_counts().reset_index(name='Number_of_Entry')
print(Entry_per_project_per_year.head())

# Sort the DataFrame by ProjectType and EntryYear to ensure proper cumulative sum calculation
Entry_per_project_per_year = Entry_per_project_per_year.sort_values(by=['ProjectType', 'EntryYear'])

# Calculate the cumulative sum of Number_of_Entry per ProjectType
Entry_per_project_per_year['Cumulative_Entry'] = Entry_per_project_per_year.groupby('ProjectType')['Number_of_Entry'].cumsum()

# Print the updated DataFrame
print(Entry_per_project_per_year.head())

# Group by ProjectType and ExitYear and count the number of exits
Exit_per_project_per_year = clean_data_df.groupby(['ProjectType','ExitYear'])['ProjectType'].value_counts().reset_index(name='Number_of_Exit')
print(Exit_per_project_per_year.head())

# Sort the DataFrame by ProjectType and EntryYear to ensure proper cumulative sum calculation
Exit_per_project_per_year = Exit_per_project_per_year.sort_values(by=['ProjectType', 'ExitYear'])

# Calculate the cumulative sum of Number_of_Entry per ProjectType
Exit_per_project_per_year['Cumulative_Exit'] = Exit_per_project_per_year.groupby('ProjectType')['Number_of_Exit'].cumsum()

# Print the updated DataFrame
print(Exit_per_project_per_year.head())

# Merge Entry and Exit DataFrames on ProjectType and Year
info_per_project = pd.merge(
    Entry_per_project_per_year,
    Exit_per_project_per_year,
    left_on=['ProjectType', 'EntryYear'],
    right_on=['ProjectType', 'ExitYear'],
    # suffixes=('_Entry', '_Exit'),
    how='left'  # Keep all entries even if no exit data exists
)


# Fill NaN values in Exit Capacity with 0 (no exits)
info_per_project['Number_of_Exit'].fillna(0, inplace=True)

# Subtract cumalitive Exit from cumalative Entry to get the remaining capacity
info_per_project['Remained'] = info_per_project.groupby('ProjectType')['Number_of_Entry'].cumsum() - info_per_project.groupby('ProjectType')['Number_of_Exit'].cumsum()

# Drop the ExitYear column if you don't need it
info_per_project.drop(columns=['ExitYear'], inplace=True)

# Rename EntryYear back to Year for clarity
info_per_project.rename(columns={'EntryYear': 'Year'}, inplace=True)

# Final Result
print(info_per_project.head())

# Sort the DataFrame by ProjectType and Year to ensure correct cumulative operations
info_per_project = info_per_project.sort_values(by=['ProjectType', 'Year'])

# Initialize Capacity column with the first year's Remained value
info_per_project['Capacity'] = info_per_project['Number_of_Entry']

# Iterate through each project type and calculate Capacity for each year
for project in info_per_project['ProjectType'].unique():
    project_df = info_per_project[info_per_project['ProjectType'] == project]
    # Start from the second row and update Capacity by adding Number_of_Entry to the previous year's Capacity
    for i in range(1, len(project_df)):
        current_index = project_df.index[i]
        previous_index = project_df.index[i - 1]
        
        # Capacity = Previous Year's Capacity + Current Year's Entry
        info_per_project.at[current_index, 'Capacity'] = (
            info_per_project.at[previous_index, 'Remained'] + 
            info_per_project.at[current_index, 'Capacity']
        )

# Print the final DataFrame with Capacity
print(info_per_project.head())

# Get the highest Number_of_Entry per project and its corresponding Year
highest_entry = info_per_project.loc[info_per_project.groupby('ProjectType')['Capacity'].idxmax(), ['ProjectType', 'Year', 'Capacity']]
print(highest_entry)
print(highest_entry.Capacity.sum())

# Save the final capacity DataFrame to a CSV file
capacity_per_project = highest_entry[['ProjectType', 'Capacity']]
print(capacity_per_project)
capacity_per_project.to_csv(f'{Save_path}Capacity_per_project_from_cleaned_data.csv', index=False) 