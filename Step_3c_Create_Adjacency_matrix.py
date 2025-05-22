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
# 'median_los_imputation_for_exit_data_last_four_occurrence'
occurrence_type = 'median_los_imputation_for_exit_data_last_four_occurrence' 

## Load data
# Load Initial entry info
Initial_entry_df = pd.read_csv(f'{File_path}initial_entry_weight_from_cleaned_data_2012_upto_2017_{occurrence_type}.csv')
print(Initial_entry_df)

# Load Existing Adjacency matrix
Adjacency_matrix_df = pd.read_csv(f'{File_path}Empirical_Adjacency_matrix_{occurrence_type}.csv')

## Data cleaning
# Change NAN values to 0
Adjacency_matrix_df = Adjacency_matrix_df.fillna(0)
print(Adjacency_matrix_df)

## Add the "Entry" row in the Adjacency matrix
# Create a new row with zeros
new_row = pd.DataFrame([[0] * (len(Adjacency_matrix_df.columns))], columns=Adjacency_matrix_df.columns)
print(new_row)

# Ensure 'Weight' column has exactly 9 values
weights = Initial_entry_df['Weight'].values  # Convert to NumPy array

# Update new_row while keeping last columns as 0
# Fill the "Entry to other projects" with the weight of the initial entry
new_row.iloc[0, 1:-1] = weights  # Assign weights to middle columns
print(new_row)

  
# Insert the new row at the desired position (index 1 for the second row)
Adjacency_matrix_df = pd.concat([Adjacency_matrix_df.iloc[:0], new_row, Adjacency_matrix_df.iloc[0:]]).reset_index(drop=True)
print(Adjacency_matrix_df)

# Add the "Entry" column. Here Entry ois denoted with 0
Adjacency_matrix_df.insert(loc=1, column='0', value=0)
print(Adjacency_matrix_df)

    
# Save the updated Adjacency matrix
Adjacency_matrix_df.to_csv(f'{File_path}Adjacency_matrix_with_entry_exit_Project_Type_{occurrence_type}.csv', index=False)

