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

median_los_per_project = pd.read_csv(f'{Save_path}median_length_of_stay_per_project.csv', index_col='ProjectType').drop(['Unnamed: 0'],axis=1)
print(median_los_per_project)


# Missing values in the 'ExitDate' column changed with the median length of stay
clean_data_exit_filled_with_median_los = clean_data_df.copy()

# Convert 'EntryDate' and 'ExitDate' to datetime
clean_data_exit_filled_with_median_los['EntryDate'] = pd.to_datetime(clean_data_exit_filled_with_median_los['EntryDate'])
clean_data_exit_filled_with_median_los['ExitDate'] = pd.to_datetime(clean_data_exit_filled_with_median_los['ExitDate'])

# Impute missing exit dates using median LOS per project
def impute_exit(row):
    '''    Function to impute missing exit dates using median length of stay per project.
    Args:
        row (Series): A row of the DataFrame.
    Returns:
        datetime: The imputed exit date.
    '''
    # Check if the exit date is missing
    if pd.isna(row['ExitDate']):
        los = median_los_per_project.get(row['ProjectType'], 78)  # fallback: 78 days
        return row['EntryDate'] + pd.Timedelta(days=los)
    return row['ExitDate']

clean_data_exit_filled_with_median_los['ExitDate'] = clean_data_exit_filled_with_median_los.apply(impute_exit, axis=1)
print(len(clean_data_exit_filled_with_median_los)) # Check the number of rows after filling missing exit dates
# Save this data to a CSV file
clean_data_exit_filled_with_median_los.to_csv(f'{Save_path}Clean_data_using_missing_exit_filled_with_median_los.csv', index=False)

# Keep only the data from 2012 onwards till 2017
clean_data_exit_filled_with_median_los_2012_onwards = clean_data_exit_filled_with_median_los[(clean_data_exit_filled_with_median_los["EntryDate"].dt.year >= 2012) & (clean_data_exit_filled_with_median_los["EntryDate"].dt.year <= 2017)]

#Extarct only the year and month from EntryDate
clean_data_exit_filled_with_median_los_2012_onwards['EntryYear'] = clean_data_exit_filled_with_median_los_2012_onwards['EntryDate'].dt.year
clean_data_exit_filled_with_median_los_2012_onwards['EntryMonth'] = clean_data_exit_filled_with_median_los_2012_onwards['EntryDate'].dt.month
print(len(clean_data_exit_filled_with_median_los_2012_onwards)) # Check the number of rows after filtering
print(len(clean_data_exit_filled_with_median_los_2012_onwards.ClientID.unique())) # Check the number of unique clients after filtering

# Save this data to a CSV file
clean_data_exit_filled_with_median_los_2012_onwards.to_csv(f'{Load_path}Clean_data_using_missing_exit_filled_with_median_los_2012_onwards.csv', index=False)

def setProjectType_to_exit(df, position):
    '''
    Function to set ProjectType to -3 for the specified occurrence of each ClientID.
    Args:
        df (DataFrame): The DataFrame containing the data.
    Returns:
        DataFrame: A DataFrame with ProjectType set to -3 for the specified occurrence.
    '''
    # Ensure the EntryDate is in datetime format
    df['EntryDate'] = pd.to_datetime(df['EntryDate'])

    # Assuming df is already sorted by ClientID and EntryDate
    df = df.sort_values(by=['ClientID', 'EntryDate'])

    # Check the position argument
    if position == 'last':
        # Get the last occurrence of each ClientID
        last_row = df.groupby('ClientID').tail(1)

        # Build new rows with ProjectType = -3 and EntryDate = ExitDate
        exit_row = last_row.copy()
        exit_row['EntryDate'] = exit_row['ExitDate']
        exit_row['ProjectType'] = -3
        # Set ExitDate = EntryDate + 30 days
        exit_row['ExitDate'] = exit_row['EntryDate'] + pd.Timedelta(days=30)
        # Append these new rows back to the filtered DataFrame
        df = pd.concat([df, exit_row], ignore_index=True)
        # Sort the DataFrame by EntryDate
        df = df.sort_values(by=['EntryDate'])
    # Keep the last 4 occurrences per client
    elif position == 'last_four':
        # Keep only the last 4 occurrences per client
        df_last_4 = df.groupby('ClientID').tail(4)

        # Identify the last occurrence per client
        last_row = df_last_4.groupby('ClientID').tail(1)

        # Build new rows with ProjectType = -3 and EntryDate = ExitDate
        exit_row = last_row.copy()
        exit_row['EntryDate'] = exit_row['ExitDate']
        exit_row['ProjectType'] = -3

        # Set ExitDate = EntryDate + 30 days
        exit_row['ExitDate'] = exit_row['EntryDate'] + pd.Timedelta(days=30)

        # Append these new rows back to the filtered DataFrame
        df = pd.concat([df_last_4, exit_row], ignore_index=True)
        # Sort the DataFrame by EntryDate
        df = df.sort_values(by=['EntryDate'])

    return df



# Set the ProjectType to -3 for the last occurrence of each client
exit_filled_with_median_los_2012_onwards_last_occurrence = setProjectType_to_exit(clean_data_exit_filled_with_median_los_2012_onwards, 'last')
print(len(exit_filled_with_median_los_2012_onwards_last_occurrence)) # Check the number of rows after filtering
print(len(exit_filled_with_median_los_2012_onwards_last_occurrence.ClientID.unique())) # Check the number of unique clients after filtering

# Save this data to a CSV file  
exit_filled_with_median_los_2012_onwards_last_occurrence.to_csv(f'{Save_path}Clean_data_using_missing_exit_filled_with_median_los_2012_onwards_-3_at_last_occurrence.csv', index=False)

# Set the ProjectType to -3 for the last and keep last four occurrence of each client
exit_filled_with_median_los_2012_onwards_last_four_occurrence = setProjectType_to_exit(clean_data_exit_filled_with_median_los_2012_onwards, 'last_four')
print(len(exit_filled_with_median_los_2012_onwards_last_four_occurrence)) # Check the number of rows after filtering
print(len(exit_filled_with_median_los_2012_onwards_last_four_occurrence.ClientID.unique())) # Check the number of unique clients after filtering
print(exit_filled_with_median_los_2012_onwards_last_four_occurrence.head(10)) # Check the first 10 rows after filtering
# Save this data to a CSV file
exit_filled_with_median_los_2012_onwards_last_four_occurrence.to_csv(f'{Save_path}Clean_data_using_missing_exit_filled_with_median_los_2012_onwards_-3_at_last_four_occurrence.csv', index=False)