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
Save_path = 'Generated files for ASONAM 2025/'

## Load data
clean_data_df = pd.read_csv(f'{Load_path}20210620_Data with feature with 60 percent or more missing data removed.csv').drop(['Unnamed: 0'],axis=1)

##population of clean data
population = len(clean_data_df['ClientID'].unique())
population # Found 6011

# Convert column to datetime type
clean_data_df['EntryDate'] = pd.to_datetime(clean_data_df['EntryDate'])
clean_data_df['ExitDate'] = pd.to_datetime(clean_data_df['ExitDate'])

# Extract only the year
clean_data_df['EntryYear'] = clean_data_df['EntryDate'].dt.year
clean_data_df['ExitYear'] = clean_data_df['ExitDate'].dt.year

clean_data_df['EntryMonth'] = clean_data_df['EntryDate'].dt.month
clean_data_df['ExitMonth'] = clean_data_df['ExitDate'].dt.month

print(clean_data_df.head())

## Entry in each month of each year
Entry_per_project_per_month = clean_data_df.groupby(['EntryYear','EntryMonth','ProjectType'])['ProjectType'].value_counts().reset_index(name='Number_of_Entry_per_Month')
print(Entry_per_project_per_month.head())
Entry_per_project_per_month.to_csv(f'{Save_path}Entry_per_project_per_month.csv', index=False) 
