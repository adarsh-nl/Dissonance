import time
from keras.models import load_model
import os
import json
import time
import flwr as fl
import numpy as np
from typing import Optional
from keras.models import load_model
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json
from keras.utils import to_categorical
from sklearn.metrics import f1_score,classification_report

#import custom scripts
from Custom_Scripts.debug import debug
from Custom_Scripts.Eval_and_CheckDrfit import convert_to_dict, find_drifted_clients
from Custom_Scripts.update_cluster import split_clusters
from Custom_Scripts.constants import BASE_PATH


def fit_config(rnd: int):

    """
    Return training configuration dictionary for each communication round.

    Args:
    - rnd (int): The communication round number.

    Returns:
    - dict: Training configuration dictionary.

    Example:
    >>> fit_config(3)
    {'num_rounds': 3}
    """
    config = {
         "num_rounds":rnd,
    }
    return config


def get_eval_fn(model, X_val, y_val):

    def evaluate(
        server_round: int,
        parameters_ndarrays: List[fl.common.NDArray],
        client_info: Dict[str, Union[bool, bytes, float, int, str]],
        message_from_client: Optional[str] = None
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:

        parameters = fl.common.ndarrays_to_parameters(parameters_ndarrays)
        model.set_weights(fl.common.parameters_to_ndarrays(parameters))

        loss, accuracy = model.evaluate(X_val, y_val)

        y_pred = model.predict(X_val)
        y_pred_labels = np.argmax(y_pred, axis=-1)
        y_val_labels = np.argmax(y_val, axis=-1)

        # Calculate F1 score for weighted average
        f1 = f1_score(y_val_labels.flatten(), y_pred_labels.flatten(), average="weighted")

        print("F1 Score:", f1)

        return loss, {"accuracy": accuracy, "F1_score": f1}

    return evaluate


def save_trained_server_model(model, comm_round, key):

    """
    Save the trained server model for a specific cohort after a communication round.

    Args:
    - model: The server model to be saved.
    - comm_round (int): The communication round number.
    - key: A unique identifier for the cohort.

    Returns:
    - None

    Example:
    >>> save_trained_server_model(my_model, 3, 1)
    """

    directory_path = BASE_PATH + 'Results\server'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    model.save(directory_path + f'\\server_model_round{comm_round}_cohort{key}.h5')
    return

def load_previous_model(comm_round, key):

    """
    Load the server model from the previous communication round for a specific cohort.

    Args:
    - comm_round (int): The current communication round number.
    - key: A unique identifier for the cohort.

    Returns:
    - Model: The loaded server model if found, or None if not found.

    Example:
    >>> loaded_model = load_previous_model(3, 1)
    """

    model_path = BASE_PATH + f'Results\server\server_model_round{comm_round-1}_cohort{key}.h5'
    if os.path.exists(model_path):
        return load_model(model_path)
    else:
        return None

def wait_for_clients_completion(comm_round, NUM_CLIENTS, timeout=45):  # Set a timeout of 45 Seconds (adjust as needed)

    """
    Wait for all clients to complete a specific communication round.

    Args:
    - comm_round (int): The communication round number.
    - NUM_CLIENTS (int): The total number of clients participating.
    - timeout (int, optional): The maximum time (in seconds) to wait for completion. Default is 45 seconds.

    Returns:
    - None

    Raises:
    - TimeoutError: If the timeout is exceeded before all clients complete.

    Example:
    >>> wait_for_clients_completion(3, 10, timeout=60)
    """

    completion_counter = 0
    start_time = time.time()

    while completion_counter < NUM_CLIENTS and (time.time() - start_time) < timeout:
        with open(BASE_PATH + 'Results\completion_status.txt', 'r') as completion_file:
            content = completion_file.read()
            completion_counter = content.count(f'completed-{comm_round}')

        if completion_counter < NUM_CLIENTS:
            print(f"Server is waiting for all {NUM_CLIENTS} clients to complete communication round {comm_round}.")
            time.sleep(30)

    if time.time() - start_time >= timeout:
        print(f"Timeout exceeded. Some clients did not update completion status for communication round {comm_round}.")

def update_status_file_as_completed(comm_round):

    """
    Update the status file to mark a communication round as completed.

    Args:
    - comm_round (int): The communication round number.

    Returns:
    - None

    Example:
    >>> update_status_file_as_completed(3)
    """

    with open(BASE_PATH + 'Results\communication_round_status.txt', 'w') as status_file:
        status_file.write(f'Communication round {comm_round} completed')

def update_status_file_as_not_completed(comm_round):

    """
    Update the status file to mark a communication round as not completed.

    Args:
    - comm_round (int): The communication round number.

    Returns:
    - None

    Example:
    >>> update_status_file_as_not_completed(3)
    """

    with open(BASE_PATH + 'Results\communication_round_status.txt', 'w') as status_file:
        status_file.write(f'Communication round {comm_round} not completed')

def create_neccessary_text_files(comm_round):

    """
    Create necessary text files for communication round.

    Args:
    - comm_round (int): The communication round number.

    Returns:
    - None

    Example:
    >>> create_neccessary_text_files(3)
    """

    #Open a status file in the server for the clients to update their status.
    debug(f"Server>>>> Commuication round: {comm_round} - Status file crated.")
    directory_path = BASE_PATH + f'Results\client_status'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    with open(directory_path + f"\\iteration_{comm_round}.txt", "w") as file:
        pass

    with open(BASE_PATH + f"Results\completion_status.txt", "w") as file:
        pass

def server_functions(comm_round, current_cluster):        

    """
    Execute server functions for a communication round.

    Args:
    - comm_round (int): The communication round number.
    - current_cluster (dict): Dictionary representing the current clustering of clients.

    Returns:
    - Tuple[List[int], List[int], dict]: Drifted clients, non-drifted clients, and updated cluster.

    Example:
    >>> drifted, non_drifted, updated_cluster = server_functions(3, {'cluster1': [1, 2], 'cluster2': [3, 4]})
    """    

    time.sleep(30) #Wait till all the clients update their status into the db
    drift_data  = convert_to_dict(comm_round)
    drifted_clients, non_drifted_clients = find_drifted_clients(drift_data)
    debug(f"Server>>>> Communication round: {comm_round}, Current cluster: {current_cluster}")

    #update the clusters based on the drifted clients
    current_cluster = split_clusters(current_cluster, drifted_clients)
    debug(f"Server>>>> Communication round: {comm_round},Updated cluster: {current_cluster}")

    return drifted_clients, non_drifted_clients, current_cluster

def get_ready_clients(clients):

    """
    Notify clients to get ready for the training process.

    Args:
    - clients (list): List of client IDs to notify.

    Returns:
    - None

    Example:
    >>> get_ready_clients([1, 2, 3])
    """

    with open(BASE_PATH + r"Scripts\\Clients\\get_ready_clients.txt", "w") as file:
        # Join the elements of the list into a string with a separator (e.g., a comma)
        clients_str = ', '.join(map(str, clients))
        file.write(clients_str)
    debug(f"Server>>>> The following clients are notified to get ready for the training process.\n{clients}")

def save_results(comm_round, cohort, history_):

    """
    Save federated learning results for a communication round.

    Args:
    - comm_round (int): The communication round number.
    - cohort (int): Cohort identifier.
    - history_ (object): Object containing centralized losses and metrics.

    Returns:
    - None

    Example:
    >>> save_results(3, 1, history_object)
    """

    if comm_round>1:
        with open(BASE_PATH + r'Results\\FL_results.json', 'a') as f:
            json.dump({
                "round": comm_round,
                "cohort":cohort,
                "losses": history_.losses_centralized,
                "metrics": history_.metrics_centralized
            }, f)
            f.write('\n')  # Add a newline to separate each iteration
    else:
        with open(BASE_PATH + r'Results\\FL_results.json', 'w') as f:
            json.dump({
                "round": comm_round,
                "cohort":cohort,
                "losses": history_.losses_centralized,
                "metrics": history_.metrics_centralized
            }, f)
            f.write('\n')  # Add a newline to separate each iteration

# Function to save the current_cluster dictionary to a file
def save_cluster_to_file(current_cluster, comm_round):

    """
    Save the current cluster configuration to a JSON file.

    Args:
    - current_cluster (dict): Dictionary representing the current cluster configuration.
    - comm_round (int): The communication round number.

    Returns:
    - None

    Example:
    >>> save_cluster_to_file({'cluster_1': [1, 2, 3], 'cluster_2': [4, 5]}, 3)
    """

    directory_path = BASE_PATH + r'Results\\Clusters'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    file_path = os.path.join(directory_path, f'current_cluster_round{comm_round}.json')
    with open(file_path, 'w') as file:
        #file.write(current_cluster)
        json.dump(current_cluster, file)

def evaluate_model_and_get_metrics(model, X_test, y_test, previous_accuracy = 0, previous_loss=0):

    """
    Evaluate the model on the test data and calculate metrics.

    Args:
    - model: Trained machine learning model.
    - X_test (numpy.ndarray): Test features.
    - y_test (numpy.ndarray): Test labels.
    - previous_accuracy (float): Accuracy from the previous evaluation.
    - previous_loss (float): Loss from the previous evaluation.

    Returns:
    - Tuple: Current loss, change in loss, current accuracy, change in accuracy.

    Example:
    >>> evaluate_model_and_get_metrics(trained_model, test_features, test_labels, 0.8, 0.2)
    """

    current_loss, current_accuracy = model.evaluate(X_test, y_test)
    delta_loss = current_loss - previous_loss
    delta_accuracy = current_accuracy - previous_accuracy
    return current_loss, delta_loss, current_accuracy, delta_accuracy

def save_metrics(current_loss, delta_loss, current_accuracy, delta_accuracy):

    """
    Save the evaluation metrics to a JSON file.

    Args:
    - current_loss (float): Current loss value.
    - delta_loss (float): Change in loss from the previous evaluation.
    - current_accuracy (float): Current accuracy value.
    - delta_accuracy (float): Change in accuracy from the previous evaluation.

    Returns:
    - None

    Example:
    >>> save_metrics(0.5, -0.1, 0.85, 0.05)
    """

    with open(BASE_PATH + r'Results\\metrics.json', 'w') as f:
        json.dump({
            "Current_loss": current_loss,
            "delta loss":delta_loss,
            "Current_accuracy": current_accuracy,
            "delta_accuracy": delta_accuracy
        }, f)
        f.write('\n')  # Add a newline to separate each iteration