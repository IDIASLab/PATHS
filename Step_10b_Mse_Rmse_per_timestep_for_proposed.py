## Libraries

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import IPython
from mpl_toolkits.mplot3d import Axes3D

# Data manipulation
import pandas as pd
import numpy as np
import sklearn
from sklearn.metrics import mean_squared_error
import scipy.stats as stats

# Get and save data
import os
from os import path

## Data Path
# Define the path to Load data from
Load_path = 'Generated files for ASONAM 2025/Multiple_simulations/Using_all_2012_2017/'
GT_path = 'Generated files for ASONAM 2025/Empirical_info/'
fig_path = 'Final_Results/Using_all_2012_2017/'

# Give name to the project types
project_types_names = ['P1','P2','P3','P4','P6','P11','P12','P13','P14', 'Exit']

# Change this to 
# 'median_los_imputation_for_exit_data_last_occurrence', 
# 'median_los_imputation_for_exit_data_last_four_occurrence', 
# as needed
occurrence_type = 'median_los_imputation_for_exit_data_last_four_occurrence' 

if occurrence_type == 'median_los_imputation_for_exit_data_last_four_occurrence':
    Load_path = 'Generated files for ASONAM 2025/Multiple_simulations/Using_last_four_occurrences/'
    fig_path = 'Final_Results/Using_last_four_occurrences/'


# # Load the error of each simulation model
simulation_df = {}
for simulation_no in range(1, 11):
    simulation_df[simulation_no] = pd.read_csv(f'{Load_path}SE_Absolute_Error_per_project_per_timestep_proposed_model{simulation_no}.csv')

# sum up the SE for each simulation per timestep per project
simulation_df[11] = pd.DataFrame()
simulation_df[11] = simulation_df[1].copy()
for i in range(2, 11):
    simulation_df[11]['SE'] += simulation_df[i]['SE']
    simulation_df[11]['Absolute_Error'] += simulation_df[i]['Absolute_Error']
# divide the sum by 10 to get the average
simulation_df[11]['MSE'] = simulation_df[11]['SE'] / 10
simulation_df[11]['Absolute_Error'] /= 10
simulation_df[11]['RMSE'] = np.sqrt(simulation_df[11]['MSE'])

mse = (simulation_df[11]['MSE'].sum())/720
print(f'MSE: {mse}')
mse_without_exit = (simulation_df[11][simulation_df[11]['Project'] != 'Exit']['MSE'].sum())/648
print(f'MSE without exit: {mse_without_exit}')
exit_mse = (simulation_df[11][simulation_df[11]['Project'] == 'Exit']['MSE'].sum())/72
print(f'MSE for exit: {exit_mse}')
rmse = (simulation_df[11]['RMSE'].sum())/720
print(f'RMSE: {rmse}')
rmse_without_exit = (simulation_df[11][simulation_df[11]['Project'] != 'Exit']['RMSE'].sum())/648
print(f'RMSE without exit: {rmse_without_exit}')
exit_rmse = (simulation_df[11][simulation_df[11]['Project'] == 'Exit']['RMSE'].sum())/72
print(f'RMSE for exit: {exit_rmse}')
mae = (simulation_df[11]['Absolute_Error'].sum())/720
print(f'MAE: {mae}')
mae_without_exit = (simulation_df[11][simulation_df[11]['Project'] != 'Exit']['Absolute_Error'].sum())/648
print(f'MAE without exit: {mae_without_exit}')
exit_mae = (simulation_df[11][simulation_df[11]['Project'] == 'Exit']['Absolute_Error'].sum())/72
print(f'MAE for exit: {exit_mae}')

# Save the averaged results
simulation_df[11].to_csv(f'{Load_path}SE_MSE_RMSE_Absolute_Error_per_project_per_timestep_proposed_model.csv', index=False)

# Plot the RMSE for each project over timesteps
plt.figure(figsize=(12, 6))
sns.lineplot(data=simulation_df[11], x='Timestep', y='RMSE', hue='Project', marker='o')

# plt.title('RMSE over Timesteps for Each Project')
plt.xlabel('Timestep (in months)', fontsize=16, labelpad=15)
plt.ylabel('RMSE', fontsize=16, labelpad=15)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)
plt.legend(title='Project', fontsize=16, title_fontsize=16)
plt.tight_layout()
plt.savefig(f'{fig_path}RMSE_per_project_per_timestep_{occurrence_type}.png')
plt.show()


#Plot the Absolute Error for each project over timesteps
plt.figure(figsize=(12, 6))
sns.lineplot(data=simulation_df[11], x='Timestep', y='Absolute_Error', hue='Project', marker='o')

# plt.title('RMSE over Timesteps for Each Project')
plt.xlabel('Timestep (in months)', fontsize=16, labelpad=15)
plt.ylabel('Absolut Error', fontsize=16, labelpad=15)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.grid(True)
plt.legend(title='Project', fontsize=16, title_fontsize=16)
plt.tight_layout()
plt.savefig(f'{fig_path}Absolute_error_per_project_per_timestep_{occurrence_type}.png')
plt.show()


## Confidence interval for MSE
mse_values = simulation_df[11]['MSE'].values

# Mean and standard error
mean_mse = np.mean(mse_values)
sem = stats.sem(mse_values)  # Standard error of the mean

# 95% confidence interval
conf_interval_mse = stats.t.interval(
    confidence=0.95,
    df=len(mse_values)-1,
    loc=mean_mse,
    scale=sem
)

print("Mean MSE:", mean_mse)

# plus minus error
plus_minus_mse = (conf_interval_mse[1] - conf_interval_mse[0]) / 2
print("Plus minus error for MSE:", plus_minus_mse)

## Confidence interval for MSE without exit
mse_values_without_exit = simulation_df[11][simulation_df[11]['Project'] != 'Exit']['MSE'].values

# Mean and standard error
mean_mse_without_exit = np.mean(mse_values_without_exit)
sem = stats.sem(mse_values_without_exit)  # Standard error of the mean

# 95% confidence interval
conf_interval_mse = stats.t.interval(
    confidence=0.95,
    df=len(mse_values_without_exit)-1,
    loc=mean_mse_without_exit,
    scale=sem
)

print("Mean MSE:", mean_mse_without_exit)

# plus minus error
plus_minus_mse = (conf_interval_mse[1] - conf_interval_mse[0]) / 2
print("Plus minus error for MSE:", plus_minus_mse)

## Confidence interval for MSE for exit
mse_values_exit = simulation_df[11][simulation_df[11]['Project'] == 'Exit']['MSE'].values
# Mean and standard error
mean_mse_exit = np.mean(mse_values_exit)
sem = stats.sem(mse_values_exit)  # Standard error of the mean
# 95% confidence interval
conf_interval_mse = stats.t.interval(
    confidence=0.95,
    df=len(mse_values_exit)-1,
    loc=mean_mse_exit,
    scale=sem
)
print("Mean MSE for exit:", mean_mse_exit)
# plus minus error
plus_minus_mse = (conf_interval_mse[1] - conf_interval_mse[0]) / 2
print("Plus minus error for MSE for exit:", plus_minus_mse)



## Confidence interval for RMSE
rmse_values = simulation_df[11]['RMSE'].values
# Mean and standard error   
mean_rmse = np.mean(rmse_values)
sem = stats.sem(rmse_values)  # Standard error of the mean
# 95% confidence interval
conf_interval_rmse = stats.t.interval(
    confidence=0.95,
    df=len(rmse_values)-1,
    loc=mean_rmse,
    scale=sem
)
print("Mean RMSE:", mean_rmse)
# plus minus error
plus_minus_rmse = (conf_interval_rmse[1] - conf_interval_rmse[0]) / 2
print("Plus minus error for RMSE:", plus_minus_rmse)

# ## Confidence interval for RMSE without exit
rmse_values_without_exit = simulation_df[11][simulation_df[11]['Project'] != 'Exit']['RMSE'].values
# Mean and standard error
mean_rmse_without_exit = np.mean(rmse_values_without_exit)
sem = stats.sem(rmse_values_without_exit)  # Standard error of the mean
# 95% confidence interval   
conf_interval_rmse = stats.t.interval(
    confidence=0.95,
    df=len(rmse_values_without_exit)-1,
    loc=mean_rmse_without_exit,
    scale=sem
)
print("Mean RMSE without exit:", mean_rmse_without_exit)
# plus minus error
plus_minus_rmse = (conf_interval_rmse[1] - conf_interval_rmse[0]) / 2
print("Plus minus error for RMSE without exit:", plus_minus_rmse)

## Confidence interval for RMSE for exit
rmse_values_exit = simulation_df[11][simulation_df[11]['Project'] == 'Exit']['RMSE'].values
# Mean and standard error
mean_rmse_exit = np.mean(rmse_values_exit)
sem = stats.sem(rmse_values_exit)  # Standard error of the mean
# 95% confidence interval
conf_interval_rmse = stats.t.interval(
    confidence=0.95,
    df=len(rmse_values_exit)-1,
    loc=mean_rmse_exit,
    scale=sem
)
print("Mean RMSE for exit:", mean_rmse_exit)
# plus minus error
plus_minus_rmse = (conf_interval_rmse[1] - conf_interval_rmse[0]) / 2
print("Plus minus error for RMSE for exit:", plus_minus_rmse)



## Confidence interval for Absolute Error
abs_error_values = simulation_df[11]['Absolute_Error'].values
# Mean and standard error
mean_abs_error = np.mean(abs_error_values)
sem = stats.sem(abs_error_values)  # Standard error of the mean
# 95% confidence interval
conf_interval_abs_error = stats.t.interval(
    confidence=0.95,
    df=len(abs_error_values)-1,
    loc=mean_abs_error,
    scale=sem
)
print("Mean Absolute Error:", mean_abs_error)
# plus minus error
plus_minus_abs_error = (conf_interval_abs_error[1] - conf_interval_abs_error[0]) / 2    
print("Plus minus error for Absolute Error:", plus_minus_abs_error)

## Confidence interval for Absolute Error without exit
abs_error_values_without_exit = simulation_df[11][simulation_df[11]['Project'] != 'Exit']['Absolute_Error'].values
# Mean and standard error
mean_abs_error_without_exit = np.mean(abs_error_values_without_exit)
sem = stats.sem(abs_error_values_without_exit)  # Standard error of the mean
# 95% confidence interval
conf_interval_abs_error = stats.t.interval(
    confidence=0.95,
    df=len(abs_error_values_without_exit)-1,
    loc=mean_abs_error_without_exit,
    scale=sem
)
print("Mean Absolute Error without exit:", mean_abs_error_without_exit)
# plus minus error
plus_minus_abs_error = (conf_interval_abs_error[1] - conf_interval_abs_error[0]) / 2
print("Plus minus error for Absolute Error without exit:", plus_minus_abs_error)

## Confidence interval for Absolute Error for exit
abs_error_values_exit = simulation_df[11][simulation_df[11]['Project'] == 'Exit']['Absolute_Error'].values
# Mean and standard error
mean_abs_error_exit = np.mean(abs_error_values_exit)
sem = stats.sem(abs_error_values_exit)  # Standard error of the mean
# 95% confidence interval
conf_interval_abs_error = stats.t.interval(
    confidence=0.95,
    df=len(abs_error_values_exit)-1,
    loc=mean_abs_error_exit,
    scale=sem
)
print("Mean Absolute Error for exit:", mean_abs_error_exit)
# plus minus error
plus_minus_abs_error = (conf_interval_abs_error[1] - conf_interval_abs_error[0]) / 2
print("Plus minus error for Absolute Error for exit:", plus_minus_abs_error)