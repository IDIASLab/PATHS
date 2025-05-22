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
Load_path = 'Generated files for ASONAM 2025/Emperical_info/'
fig_path = 'Final_Results/'

# Change this to 
# 'median_los_imputation_for_exit_data_last_occurrence', 
# 'median_los_imputation_for_exit_data_last_four_occurrence', 
# as needed
occurrence_type = 'median_los_imputation_for_exit_data_last_four_occurrence' 

# ## Load data
if occurrence_type == 'median_los_imputation_for_exit_data_last_occurrence':
    ground_truth_from2012 = pd.read_csv(f'{Load_path}Clean_data_using_missing_exit_filled_with_median_los_2012_onwards_-3_at_last_occurrence.csv')
elif occurrence_type == 'median_los_imputation_for_exit_data_last_four_occurrence':
    ground_truth_from2012 = pd.read_csv(f'{Load_path}Clean_data_using_missing_exit_filled_with_median_los_2012_onwards_-3_at_last_four_occurrence.csv')
print(ground_truth_from2012.head())

# Ensure EntryDate and ExitDate are datetime objects
ground_truth_from2012['EntryDate'] = pd.to_datetime(ground_truth_from2012['EntryDate'])
ground_truth_from2012['ExitDate'] = pd.to_datetime(ground_truth_from2012['ExitDate'])

# Calculate the length of stay for each client
# Group by ClientID and compute the min EntryDate and max ExitDate
date_range = ground_truth_from2012.groupby('ClientID').agg({
    'EntryDate': 'min',
    'ExitDate': 'max'
})

# Calculate the duration in months (approximate: days / 30)
date_range['Duration_Months'] = (date_range['ExitDate'] - date_range['EntryDate']).dt.days / 30

# round to nearest month
date_range['Duration_Months'] = np.floor(date_range['Duration_Months'])

print(date_range.head())

# Calculate the frequency per length of stay
Length_of_stay_df = date_range.groupby('Duration_Months').size().reset_index(name='Frequency')
#Rename columns
Length_of_stay_df.rename(columns={'Duration_Months': 'Length_of_stay'}, inplace=True)

# trim the dataframe to remove rows with Length_of_stay < 73
Length_of_stay_df = Length_of_stay_df[Length_of_stay_df['Length_of_stay'] < 73].reset_index(drop=True)

# Save the dataframe to a CSV file
Length_of_stay_df.to_csv(f'{Load_path}Ground_truth_Length_of_stay_{occurrence_type}_exit.csv', index=True)