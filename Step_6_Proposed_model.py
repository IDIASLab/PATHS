## Libraries
# Model design
import agentpy as ap
import networkx as nx
import random

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import IPython
from mpl_toolkits.mplot3d import Axes3D

# Data manipulation
import pandas as pd
import numpy as np
from scipy.stats import entropy

# Get and save data
import os
from os import path

## Data Path
# Define the path to Load data from
Load_path = 'Generated files for ASONAM 2025/Empirical_info/'
Save_path = 'Generated files for ASONAM 2025/Multiple_simulations/Using_all_2012_2017/'
fig_path = 'Final_Results/Using_all_2012_2017/'

# Change this to 
# 'median_los_imputation_for_exit_data_last_occurrence', 
# 'median_los_imputation_for_exit_data_last_four_occurrence', 
#  as needed
occurrence_type = 'median_los_imputation_for_exit_data_last_occurrence' 

## Load data
Adjacency_matrix_df = pd.read_csv(f'{Load_path}Adjacency_matrix_with_entry_exit_Project_Type.csv')
Highest_Capacity_per_project_df = pd.read_csv(f'{Load_path}Capacity_per_project_from_cleaned_data.csv')

Per_timestep_transition_median_los_imputation_for_exit_data_last_occurrence_df = pd.read_csv(f'{Load_path}Entry_per_timestep_per_project_from_2012_using_median_los_imputation_for_exit_data_last_occurrence.csv')
Per_timestep_transition_median_los_imputation_for_exit_data_last_four_occurrence_df = pd.read_csv(f'{Load_path}Entry_per_timestep_per_project_from_2012_using_median_los_imputation_for_exit_data_last_four_occurrence.csv')
Per_timestep_new_entry = pd.read_csv(f'{Load_path}Entry_per_timestep_from_2012_using_first_occurrence.csv')


if occurrence_type == 'median_los_imputation_for_exit_data_last_four_occurrence':
      Per_timestep_transition_df = pd.concat([Per_timestep_transition_median_los_imputation_for_exit_data_last_four_occurrence_df, Per_timestep_new_entry], ignore_index=True) 
      Save_path = 'Generated files for ASONAM 2025/Multiple_simulations/Using_last_four_occurrences/'
      fig_path = 'Final_Results/Using_last_four_occurrences/'
elif occurrence_type == 'median_los_imputation_for_exit_data_last_occurrence':
      Per_timestep_transition_df = pd.concat([Per_timestep_transition_median_los_imputation_for_exit_data_last_occurrence_df, Per_timestep_new_entry], ignore_index=True) 

Per_timestep_transition_df = Per_timestep_transition_df.sort_values(by=['Timestep', 'ProjectType'])
# Save the dataframe to a CSV file 
Per_timestep_transition_df.to_csv(f'{Load_path}Entry_per_timestep_per_project_from_2012_using_{occurrence_type}_with_entry.csv', index=False)
Per_timestep_transition_df = Per_timestep_transition_df[Per_timestep_transition_df['Timestep'] < 74]



# Total number of homeless people
Number_of_homeless_people = 5993

## Data Manipulation
# Get the highest Number_of_Entry per project 
Capacity_per_project_df = Highest_Capacity_per_project_df[['ProjectType','Capacity']]

# Add Entry and Exit project types
first_row = pd.DataFrame({'ProjectType': [0], 'Capacity': [Number_of_homeless_people]})
Capacity_per_project_df = pd.concat([first_row, Capacity_per_project_df], ignore_index=True)
last_row = pd.DataFrame({'ProjectType': [-3], 'Capacity': [Number_of_homeless_people]})
Capacity_per_project_df = pd.concat([Capacity_per_project_df, last_row], ignore_index=True)


# Project Types in the dataset
project_types_info = Adjacency_matrix_df['Source/Target'].unique()


# Give name to the project types
project_types_names = ['Entry','P1','P2','P3','P4','P6','P11','P12','P13','P14', 'Exit']


## Model design
# Define Projects class
class Projects:
    """ A Project represented as a node in the network. """

    def __init__(self, model, project_type, capacity):
        self.model = model  # Reference to the model
        self.project_type = project_type  # Project type
        self.capacity = capacity  # Maximum capacity of the project
        self.current_population = 0  # Current number of joined agents

    def has_capacity(self):
        """ Check if the project can accept more individuals. """
        return self.current_population < self.capacity

    def enter_individual(self, agent):
        """ Entry of an individual into the project. """
        self.current_population += 1
        self.model.network.graph.add_edge(agent, self)

    def exit_individual(self, agent):
        """ Exit of an individual from the project. """
        self.current_population -= 1
        self.model.network.graph.remove_edge(agent, self)
    


# Define Agents class
class HomelessPerson(ap.Agent):
    """ An agent representing a homeless person. """
    def setup(self):
        """ Initialize variables at agent creation. """
        self.projectType = 0  # 0 = Not in any project
        self.project_history = []  # History of project types
        self.timestep_history = []  # History of timestep
    
    def normalize_probabilities(self, probabilities):
        """ Normalize probabilities to sum to 1. """
        total_prob = sum(probabilities)
        return [prob / total_prob for prob in probabilities]

    def assign_state(self, neighbors, probabilities, timestep_entries_project_type):
        """ Assign a new state based on a list of neighbors and probabilities. """
        attempts = 0
        max_attempts = len(neighbors)  # Maximum number of attempts to find an available project

        # Normalize the probabilities to sum to 1
        probabilities = self.normalize_probabilities(probabilities)
        # Initialization of next_state
        next_state = self.projectType
               
        while(attempts < max_attempts):
            next_state = np.random.choice(neighbors, p=probabilities)
            
            # Find an available project of the next state
            available_project = next((proj for proj in self.model.projects if proj.project_type == next_state and proj.has_capacity()), None)
            # print(f'Available project: {available_project}')
            
            # If available project is found, return the next state
            if available_project is not None:
                return next_state
           
            
            # Exclude next_state from neighbors and probabilities
            filtered_neighbors = [neighbor for neighbor in neighbors if neighbor != next_state]
            filtered_probabilities = [probabilities[i] for i in range(len(neighbors)) if neighbors[i] != next_state]
            
            # # Check if there are any neighbors left
            if not filtered_neighbors:
                print("No more neighbors to choose from.")
                return None     # Return None explicitly when no neighbors are left

            # # Normalize the probabilities to sum to 1
            filtered_probabilities = self.normalize_probabilities(filtered_probabilities)
            # # Update neighbors and probabilities
            neighbors = filtered_neighbors
            probabilities = filtered_probabilities
            attempts += 1
        return self.projectType     # Return None if no valid state is found after max attempts
                
    def transition_between_projects(self, current_timestep, timestep_entries_project_type, timestep_transition_count):
        """ Transition to a project based on probability matrix. """

        transition_matrix = self.model.transition_matrix  # Access the 11x12 matrix
        # Get transition probabilities for current state
        # Choose next state based on probabilities
        current_state = self.projectType

        if (current_state != -3  # If not in Exit
            ):
            if timestep_entries_project_type !=0:
                # Find rows where column 0 has value same as current state
                
                checking_rows = transition_matrix[transition_matrix[:, 0] == current_state]

                # # Extract columns 1 to 11 if such a row exists
                if checking_rows.shape[0] > 0:
                    probabilities = checking_rows[:, 1:12].flatten()  # Extract columns 1 to 11

                # Get indices where probabilities are greater than 0
                valid_indices = np.where(probabilities > 0)[0]
                # Filter probabilities to keep only valid ones
                valid_probabilities = probabilities[valid_indices]
                valid_neighbors = project_types_info[valid_indices]
                

                if len(valid_neighbors) == 0:
                    return
                next_state = self.assign_state(valid_neighbors, valid_probabilities,timestep_entries_project_type)
                # print(f'Next state: {next_state}')
                
                if next_state is None:
                    return  # Stop transition if no valid state found
            else:
                # If no valid neighbors, return current state
                next_state = current_state

            # Find the corresponding project node
            current_project = next(proj for proj in self.model.projects if proj.project_type == current_state)
            current_project.exit_individual(self)

            new_project = next(proj for proj in self.model.projects if proj.project_type == next_state)
            new_project.enter_individual(self)
            self.projectType = new_project.project_type
            self.project_history.append(self.projectType)
            self.timestep_history.append(current_timestep)
            print(f"Agent {self} moved from {current_state} project type to {next_state} project type")
            timestep_transition_count[next_state] += 1
                
                
class TransitionModel(ap.Model):

    def setup(self):
        # Create a directed network
        graph = nx.DiGraph()

        # Create agents and network
        # Start with an empty agent list
        self.agents = ap.AgentList(self, [])  # Empty AgentList
        self.network = self.agents.network = ap.Network(self, graph)
        self.network.add_agents(self.agents, self.network.nodes)

         # Define adjacency matrix (transition probabilities)
        self.transition_matrix = self.p.transition_matrix

        # Define per_timestep_transition matrix
        self.per_timestep_transition = self.p.per_timestep_transition

        # Add all the projects to the graph
        self.projects = [Projects(self, project_type=pt, capacity = cap) for pt, cap in zip(self.p.project_types, self.p.project_capacity)]
        for project in self.projects:
            self.network.add_node(project)

        num_rows, num_cols = self.transition_matrix.shape
        
        for i, project_i in zip(range(num_rows),self.projects):  # Rows represent current states
            for j,project_j in zip(range(num_cols),self.projects):  # Columns 1-11 represent next states
                prob = self.transition_matrix[i, j+1]
                if prob > 0:
                    self.network.graph.add_edge(project_i, project_j, weight=prob)  # j-1 ensures correct indexing

        
        # timestep
        self.timestep = self.p.timestep
        self.timestep_transition_counts_from_simulation = {pt: 0 for pt in project_types_info}
    
    def process_transitions_per_timestep(self):
        """ Handle new entries and transitions based on adjacency matrix. """
        # print(self.timestep)
        timestep = 73 - self.timestep  # Reverse index to match timestep data
        print(f"\nTime step: {timestep}")
        timestep_entries = self.per_timestep_transition[self.per_timestep_transition['Timestep'] == timestep]
        
        # print("Current number of agents in the model:", len(self.agents))
        for _, entry in timestep_entries.iterrows():
            project_type = entry['ProjectType']
            num_entries = int(entry['Number_of_Entry_per_Month'])
            print(f"Project type: {int(project_type)}, Number of entries: {num_entries}")

            eligible_agents = []
            noneligible_agents = []
            agents_to_transition = []
            if project_type == 0:  # If it's the entry project
                # Create new agents
                new_agents = ap.AgentList(self, num_entries, HomelessPerson)
                new_agents.setup() # Initialize variables like projectType, history

                # Assign initial project state and add to network
                for agent in new_agents:
                    # agent.timestep = timestep

                    # Find project and attempt entry
                    entry_project = next(p for p in self.projects if p.project_type == project_type)
                    entry_project.enter_individual(agent)
                    agent.projectType = entry_project.project_type
                    agent.project_history.append(agent.projectType)
                    agent.timestep_history.append(timestep)
                    self.agents.append(agent)  # Add agent to the model's master list
                    self.network.add_agents([agent], self.network.nodes)  # Add agent to the network
                    eligible_agents.append(agent)
            else:
                # Select agents eligible for transition (could be from any project type)
                for agent in self.agents:
                    if agent.projectType != -3: # If not in Exit
                        checking_rows = self.transition_matrix[self.transition_matrix[:, 0] == agent.projectType]
                        # # Extract columns 1 to 11 if such a row exists
                        if checking_rows.shape[0] > 0:
                            probabilities = checking_rows[:, 1:12].flatten()  # Extract columns 1 to 11
                        # Get indices where probabilities are greater than 0
                        valid_indices = np.where(probabilities > 0)[0]
                        # Filter probabilities to keep only valid ones
                        neighbors = project_types_info[valid_indices]
                        if int(project_type) in neighbors:
                            eligible_agents.append(agent)
                        else: 
                            noneligible_agents.append(agent)

            
            # Randomly select up to max_transitions agents
            if(len(eligible_agents) > 0):
                sampling_agent = random.sample(eligible_agents, min(len(eligible_agents), int(num_entries)))
                agents_to_transition.extend(sampling_agent)
            
            # If number of agents doesn't meet
            if len(agents_to_transition) < int(num_entries):
                if len(noneligible_agents) > 0:
                    remaining_agents = random.sample(noneligible_agents, min(len(noneligible_agents), int(num_entries) - len(agents_to_transition)))
                    agents_to_transition.extend(remaining_agents)
            
            for agent in agents_to_transition:
                agent.transition_between_projects(timestep, project_type, self.timestep_transition_counts_from_simulation)

    def update(self):
        """ Record variables after setup and each step. """
        # Count the number of agents in each Project type
        timestep = 73 - self.timestep
        # Count number of transitions into each project type *this timestep only*
        transition_counts = self.timestep_transition_counts_from_simulation

        for i, state in zip(project_types_info, project_types_names):
            count = transition_counts[i]
            self[state] = count
            self.record(state)
        
        self.timestep_transition_counts_from_simulation = {pt: 0 for pt in project_types_info}
        self.timestep -= 1
        if self.timestep == 0:
            self.stop()

    def step(self):
        """ Define agent movement at each time step. """
        self.process_transitions_per_timestep()

    def end(self):
        """ Record final state at end of simulation. """
        self.report("Final Project Distribution", {f"{state}": self[state] for state in project_types_names})
        # Save the final network for access
        self.final_network = self.network.graph

        # Build a list of dicts (one per agent)
        agent_histories = [
            {
                "AgentID": i,
                "Project_History": np.array(agent.project_history, dtype=int),
                "Project_History_String": ",".join(str(x) for x in agent.project_history),
                "Timestep_History": np.array(agent.timestep_history, dtype=int),
                "Timestep_History_String": ",".join(str(x) for x in agent.timestep_history),
                "Final_Project_Type": agent.projectType
            }
            for i, agent in enumerate(self.agents)
        ]

        # Convert to a DataFrame
        history_df = pd.DataFrame(agent_histories)

        # Save to CSV
        history_df.to_csv(f"{Save_path}agent_histories_{(Number_of_homeless_people/1000)}k_with_{occurrence_type}_exit_sim_{simulation_no}.csv", index=False)


def homeless_3dplot(results):
    """ 3D plot of people's condition over time. """
    
    # Convert recorded results to a DataFrame
    df = pd.DataFrame(results)

    # Reset index to include the timestep as a column
    df = df.reset_index()
    
    # Set up 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create a color palette
    colors = sns.color_palette("tab10", len(project_types_names))

    # Skip the first row
    df = df.iloc[1:]

    # Plot each project type over timesteps
    for i, project in enumerate(project_types_names):
        ax.plot(df["t"], [i] * len(df), df[project], color=colors[i], label=project)

    # Axis labels
    ax.set_xlabel("Timestep (in months)", fontsize=16, labelpad=15)
    ax.set_ylabel("Project Type", fontsize=16, labelpad=15)
    ax.set_zlabel("Population Frequency", fontsize=16, labelpad=15)

    # Set y-ticks to project types names
    ax.set_yticks(range(len(project_types_names)))
    ax.set_yticklabels(project_types_names, fontsize=16)
    ax.tick_params(axis='x', labelsize=16) 
    ax.tick_params(axis='z', labelsize=16)

    # Legend
    ax.legend(loc="upper left", bbox_to_anchor=(1.1, 1), fontsize=14)
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(f'{fig_path}Population_over_projects_per_month_timestep_{(Number_of_homeless_people/1000)}k_with_{occurrence_type}_exit_sim_{simulation_no}.png')
    plt.close("all")

# Simulation no.
simulation_no = 0

for i in range(1,11):
    simulation_no = i

    # Define parameters
    parameters = {
        'population': Number_of_homeless_people,             # Total population in the clean data
        'project_types': project_types_info,  # Project types in the dataset
        'project_capacity': Capacity_per_project_df['Capacity'].to_numpy(), # Maximum patients each project can have
        'transition_matrix': Adjacency_matrix_df.to_numpy(),  # Transition matrix from the dataset
        'timestep': 74,  # Number of time steps
        'per_timestep_transition': Per_timestep_transition_df,  # Number of transitions per time step
    }
    
    # Run Model
    model = TransitionModel(parameters)
    results = model.run()           

    ## Visualization


    # Extract recorded variables
    data = results.variables.TransitionModel

    # Save the recorded data to a CSV file
    data_df = pd.DataFrame(data)
    data_df.to_csv(f'{Save_path}TransitionModel_results_{(Number_of_homeless_people/1000)}k_with_{occurrence_type}_exit_sim_{simulation_no}.csv', index=True)

    homeless_3dplot(data)
