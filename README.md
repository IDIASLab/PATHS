# PATHS
Agent-Based Modeling of Homelessness Pathways

## Introduction
PATHS is a research project focused on simulating and analyzing homelessness pathways using agent-based modeling techniques. The goal is to better understand the dynamics of homelessness.

## Features
- Agent-based simulation of homelessness pathways.
- Configurable parameters for various scenarios.
- Data visualization and analysis tools.

## Installation
To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/PATHS.git
   ```
2. Navigate to the project directory:
   ```bash
   cd PATHS
   ```
3. Create a virtual environment (optional but recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate     # On Windows
   ```
4. Install the required libraries and packages:
   ```bash
   pip install pandas numpy scipy matplotlib seaborn ipython agentpy networkx
   ```
   These libraries include:
   - `pandas`: For data manipulation and analysis.
   - `numpy`: For numerical computations.
   - `scipy`: For statistical functions and special mathematical operations.
   - `matplotlib`: For data visualization.
   - `seaborn`: For advanced data visualization.
   - `ipython`: For interactive Python sessions.
   - `agentpy`: For agent-based modeling and simulation.
   - `networkx`: For creating and analyzing complex networks.

5. If additional dependencies are listed in a `requirements.txt` file, install them as well:
   ```bash
   pip install -r requirements.txt
   ```

### Additional Notes:
- Ensure you have Python 3.8 or higher installed. You can check your Python version with:
   ```bash
   python --version
   ```
- If `pip` is not installed, you can install it by following the instructions [here](https://pip.pypa.io/en/stable/installation/).

## Usage
Once the installation is complete, follow these steps to run the project:

1. **Prepare the Dataset**:
   - Ensure the `toy_homeless_dataset.csv` file is available in the project directory. This file is required as input for the simulation. Change the directories in the code as necessary.

2. **Run the Code**:
   - The project is organized into multiple Python files, each corresponding to specific steps in the simulation process. Follow the step numbers in the filenames to execute the scripts in the correct order. For example:
     ```bash
     python Step_0a_Ground_Truth_Trajectory_per_client_Adjacency_matrix.py
     python Step_0b_Ground_Truth_data_prepare.py
     python Step_0c_Ground_truth_length_of_stay_calculation.py
     ```

3. **Create Necessary Folders**:
   - During execution, the scripts may require specific folders to save output files (e.g., CSV files or plots). Create these folders as needed and change the directories in the code accordingly. For example:
     ```bash
     mkdir Empirical_info
     mkdir Multiplu_simulations
     ```

4. **Analyze the Results**:
   - The output files (e.g., processed data, simulation results, and visualizations) will be saved in the designated folders. Use tools like pandas, matplotlib, or seaborn to further analyze the results.

5. **Customize the Workflow**:
   - Modify the scripts to adjust parameters, change input data, or customize the simulation and visualization process.

## Contributing
Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes and push the branch:
   ```bash
   git commit -m "Description of changes"
   git push origin feature-name
   ```
4. Open a pull request.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
For questions or collaboration, please contact 'ntasnim@albany.edu'.
