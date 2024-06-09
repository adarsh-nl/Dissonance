import subprocess
import os

from Custom_Scripts.constants import script_path, virtualenv_path

def trigger_server(FL_round, NUM_CLIENTS, alg, current_cluster, Use_Model_Parameters_for_cohorting):
    """
    Triggers the server script with specified parameters received from the user via UI.

    Parameters:
    - FL_round (int): The federated learning round.
    - NUM_CLIENTS (int): The number of clients participating in the federated learning.
    - alg (str): The algorithm to be used.
    - current_cluster (str): The current cluster for server deployment.
    - Use_Model_Parameters_for_cohorting (bool): Flag indicating whether to use model parameters for cohorting.

    Note:
    - The virtual environment is activated before running the server script.

    Example:
    ```
    trigger_server(5, 10, 'fedavg', 'cluster_1', True)
    ```

    """
    # Activate the virtual environment
    activate_script_path = os.path.join(virtualenv_path, 'Scripts', 'activate')
    
    # Pass parameters to the server script
    activate_command = f'"{activate_script_path}" && python "{script_path}" {FL_round} {NUM_CLIENTS} "{alg}" {Use_Model_Parameters_for_cohorting} "{current_cluster}"'
    
    # Run the command
    subprocess.run(activate_command, shell=True)
