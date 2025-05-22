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

# Give name to the project types
project_types_names = ['P1','P2','P3','P4','P6','P11','P12','P13','P14', 'Exit']

# occurence_type
# Change this to 
# 'median_los_imputation_for_exit_data_last_occurrence', 
# 'median_los_imputation_for_exit_data_last_four_occurrence', 
# as needed
occurrence_type = 'median_los_imputation_for_exit_data_last_four_occurrence' 
# occurrence_type = 'median_los_imputation_for_exit_data_last_occurrence'  

# Total number of homeless people
Number_of_homeless_people = 5993

# Load the ground truth and simulation data
ground_truth_df = pd.read_csv(f'{GT_path}Ground_truth_Median_los_imputation_for_exit_data_last_occurrence_over_timestep.csv')
print(ground_truth_df.head())

if occurrence_type == 'median_los_imputation_for_exit_data_last_four_occurrence':
    Load_path = 'Generated files for ASONAM 2025/Multiple_simulations/Using_last_four_occurrences/'
    ground_truth_df = pd.read_csv(f'{GT_path}Ground_truth_Median_los_imputation_for_exit_data_last_four_occurrence_over_timestep.csv')

# Copy df and Drop the 't' column if exists
gt = ground_truth_df.copy().iloc[:-1]
gt.set_index('Timestep', inplace=True)
gt = gt.drop(columns=['Entry'])

final_mse_rmse = pd.DataFrame(columns=['Model', 'MSE', 'MSE_except_exit',
                                       'RMSE', 'RMSE_except_exit',
                                        'Absolute_Error', 'Absolute_Error_except_exit',
                                        'MSE_exit', 'Absolute_Error_exit'])

for simulation_no in range(1, 11):
    simulation_df = pd.read_csv(f'{Load_path}TransitionModel_results_{(Number_of_homeless_people/1000)}k_with_{occurrence_type}_exit_sim_{simulation_no}.csv')

    sim = simulation_df.copy().iloc[1:73]
    sim = sim.drop(columns=['t','Entry'])

    # print(sim.head())
    # print(gt.head())

    # Initialize dictionary to store results
    results = []

    # Calculate MSE and RMSE for each project for each timestep
    for timestep in sim.index:
        for project in project_types_names:
            y_true = gt.loc[timestep, project]
            y_pred = sim.loc[timestep, project]
            se = (y_true - y_pred)**2
            absolute_error = np.abs(y_true - y_pred)
            results.append({
                'Timestep': timestep,
                'Project': project,
                'SE': se,
                'Absolute_Error': absolute_error
            })


    # Convert to DataFrame
    se_df = pd.DataFrame(results)

    print(se_df)

    # Save the results
    se_df.to_csv(f'{Load_path}SE_Absolute_Error_per_project_per_timestep_proposed_model{simulation_no}.csv', index=False)

    project_wise_se = se_df.groupby('Project')['SE'].sum().reset_index()
    # print(project_wise_se)
    project_wise_abs_error = se_df.groupby('Project')['Absolute_Error'].sum().reset_index()

    mse_per_project = project_wise_se['SE']/ 72
    # print(mse_per_project)
    # avg_mse = mse_per_project.sum()/ 10
    avg_mse = se_df['SE'].sum()/ 720
    # print(f'Average MSE: {avg_mse}')
    avg_rmse = np.sqrt(avg_mse)
    # print(f'Average RMSE: {avg_rmse}')
    abs_error_per_project = project_wise_abs_error['Absolute_Error']/ 72
    avg_abs_error = abs_error_per_project.sum()/ 10

    project_wise_se_except_exit = project_wise_se[project_wise_se['Project'] != 'Exit']
    Exit_se = project_wise_se[project_wise_se['Project'] == 'Exit']
    project_wise_abs_error_except_exit = project_wise_abs_error[project_wise_abs_error['Project'] != 'Exit']
    Exit_abs_error = project_wise_abs_error[project_wise_abs_error['Project'] == 'Exit']

    mse_per_project_except_exit = project_wise_se_except_exit['SE']/ 72
    # print(mse_per_project)
    avg_mse_except_exit = mse_per_project_except_exit.sum()/ 9
    # print(f'Average MSE: {avg_mse}')
    avg_rmse_except_exit = np.sqrt(avg_mse_except_exit)
    # print(f'Average RMSE: {avg_rmse}')
    mse_exit = Exit_se['SE'].sum()/ 72
    rmse_exit = np.sqrt(mse_exit)

    abs_error_per_project_except_exit = project_wise_abs_error_except_exit['Absolute_Error']/ 72
    avg_abs_error_except_exit = abs_error_per_project_except_exit.sum()/ 9
    abs_error_exit = Exit_abs_error['Absolute_Error'].sum()/ 72

    # new row
    new_row = pd.DataFrame([{'Model': f'Sim {simulation_no}', 'MSE': avg_mse, 'MSE_except_exit': avg_mse_except_exit, 
                                            'RMSE': avg_rmse, 'RMSE_except_exit': avg_rmse_except_exit, 
                                            'Absolute_Error': avg_abs_error, 'Absolute_Error_except_exit': avg_abs_error_except_exit,
                                            'MSE_exit': mse_exit, 'RMSE_exit':rmse_exit, 'Absolute_Error_exit': abs_error_exit}])

    final_mse_rmse = pd.concat([final_mse_rmse, new_row], ignore_index=True)

    #Proposed model pointwise SE and Absolute Error
    if(simulation_no == 1):
        proposed_model = pd.DataFrame()
        proposed_model = se_df.copy()
    else:
        proposed_model['SE'] += se_df['SE']
        proposed_model['Absolute_Error'] += se_df['Absolute_Error']

# Pointwise MSE and RMSE for the proposed model
proposed_model['MSE'] = proposed_model['SE'] / 10
proposed_model['MAE'] = proposed_model['Absolute_Error'] / 10
proposed_model['RMSE'] = np.sqrt(proposed_model['MSE'])

# Final MSE, RMSE and MAE for the proposed model
proposed_mse = (proposed_model['MSE'].sum())/720
proposed_rmse = (proposed_model['RMSE'].sum())/720
proposed_abs_error = (proposed_model['MAE'].sum())/ 720

proposed_mse_except_exit = proposed_model[proposed_model['Project']!= 'Exit']['MSE'].sum()/ (72*9)
proposed_rmse_except_exit = proposed_model[proposed_model['Project']!= 'Exit']['RMSE'].sum()/ (72*9)
proposed_abs_error_except_exit = proposed_model[proposed_model['Project']!= 'Exit']['MAE'].sum()/ (72*9)
proposed_mse_exit = proposed_model[proposed_model['Project'] == 'Exit']['MSE'].sum()/ 72
proposed_rmse_exit = proposed_model[proposed_model['Project'] == 'Exit']['RMSE'].sum()/ 72
proposed_abs_error_exit = proposed_model[proposed_model['Project'] == 'Exit']['MAE'].sum()/ 72

# New row for the proposed model results
new_row = pd.DataFrame([{'Model': 'Proposed Model', 'MSE': proposed_mse, 'MSE_except_exit': proposed_mse_except_exit, 
                                            'RMSE': proposed_rmse, 'RMSE_except_exit': proposed_rmse_except_exit, 
                                            'Absolute_Error': proposed_abs_error, 'Absolute_Error_except_exit': proposed_abs_error_except_exit,
                                            'MSE_exit': proposed_mse_exit, 'RMSE_exit': proposed_rmse_exit, 'Absolute_Error_exit': proposed_abs_error_exit}])
# Append the new row to the final DataFrame
final_mse_rmse = pd.concat([final_mse_rmse,new_row], ignore_index=True)

# Save the final results
final_mse_rmse.to_csv(f'{Load_path}MSE_RMSE_Abs_error_per_timestep_for_all_simulation_{occurrence_type}.csv', index=False)


## Confidence interval for MSE
mse_values = final_mse_rmse['MSE'].values

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

print("95% CI for MSE:", conf_interval_mse)

# plus minus error
plus_minus_mse = (conf_interval_mse[1] - conf_interval_mse[0]) / 2
print("Plus minus error for MSE:", plus_minus_mse)

## Confidence interval for RMSE
rmse_values = final_mse_rmse['RMSE'].values
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
print("95% CI for RMSE:", conf_interval_rmse)
# plus minus error
plus_minus_rmse = (conf_interval_rmse[1] - conf_interval_rmse[0]) / 2
print("Plus minus error for RMSE:", plus_minus_rmse)

## Confidence interval for Absolute Error
abs_error_values = final_mse_rmse['Absolute_Error'].values
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
print("95% CI for Absolute Error:", conf_interval_abs_error)
# plus minus error
plus_minus_abs_error = (conf_interval_abs_error[1] - conf_interval_abs_error[0]) / 2    
print("Plus minus error for Absolute Error:", plus_minus_abs_error)