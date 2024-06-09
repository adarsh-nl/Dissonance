import sys
sys.path.append('../')

from Custom_Scripts.constants import BASE_PATH
from Custom_Scripts.update_cluster import split_clusters

from keras.models import load_model
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_score
from tqdm import tqdm

import time
import os

def create_cohorts(NUM_COHORTS, cluster_labels, client_ids):
    """
    Create cohorts based on a given number of cohorts and cluster labels.

    Parameters:
    - NUM_COHORTS (int): The number of cohorts to create. Must be a positive integer.
    - cluster_labels (list): Cluster labels assigned to each client.
    - client_ids (list): List of client IDs.

    Returns:
    - cohorts (dict): A dictionary where keys represent cohort numbers (1 to NUM_COHORTS)
                     and values are lists containing client IDs belonging to each cohort.

    Raises:
    - ValueError: If NUM_COHORTS is not a positive integer,
                  or if lengths of cluster_labels and client_ids do not match.

    Example:
    >>> create_cohorts(3, [1, 2, 3, 1, 2, 3, 1, 2, 3], [101, 102, 103, 201, 202, 203, 301, 302, 303])
    {1: [101, 201, 301], 2: [102, 202, 302], 3: [103, 203, 303]}
    """

    if not isinstance(NUM_COHORTS, int) or NUM_COHORTS <= 0:
        raise ValueError("NUM_COHORTS must be a positive integer")

    if len(cluster_labels) != len(client_ids):
        raise ValueError("Lengths of cluster_labels and client_ids must match")

    cohorts = {}

    for i in range(NUM_COHORTS):
        cohort_members = [client_ids[j] for j in range(len(client_ids)) if cluster_labels[j] == i]
        cohorts[i + 1] = cohort_members

    return cohorts


def get_weights(loaded_model):

    """
    Get the flattened weights of each layer from a loaded Keras model.

    Parameters:
    - loaded_model: Keras model object.

    Returns:
    - np.ndarray: Flattened weights of all layers concatenated.
    """

    # Get the weights of each layer
    model_weights = []
    for layer in loaded_model.layers:
        layer_weights = layer.get_weights()
        model_weights.extend([w.flatten() for w in layer_weights])

    # Concatenate all flattened weights
    model_weights_flat = np.concatenate(model_weights)
    return model_weights_flat

def find_optimal_cohorts(weights):
    """
    Find the optimal number of cohorts based on the silhouette score.

    Parameters:
    - weights (numpy.array): Transformed weights of client models.

    Returns:
    - int: Optimal number of cohorts.
    """
    best_score = -1
    best_num_cohorts = -1

    for num_cohorts in range(2, min(11, len(weights))):  # Adjusted the range
        kmeans = KMeans(n_clusters=num_cohorts, init='k-means++', max_iter=500)
        kmeans.fit(weights)
        labels = kmeans.labels_
        score = silhouette_score(weights, labels)
        print(f'Cohort: {num_cohorts}: Silhouette_score: {score}')
        if score > best_score:
            best_score = score
            best_num_cohorts = num_cohorts

    print(f'Best NUM_COHORTS: {best_num_cohorts}')
    return best_num_cohorts

def load_client_model(model_path):
    """
    Load a Keras model from the specified path and retrieve its weights.

    Parameters:
    - model_path (str): The file path to the Keras model file.

    Returns:
    - numpy.ndarray: The weights of the loaded Keras model.

    Raises:
    - FileNotFoundError: If the model file is not found at the specified path.
    - OSError: If an operating system error occurs during model loading.

    Notes:
    - This function continuously attempts to load the model and retrieve its weights.
    - If an error occurs (FileNotFoundError or OSError), it prints an error message,
      waits for 10 seconds, and retries the loading process.
    """
    while True:
        try:
            model = load_model(model_path)
            weights = get_weights(model)
            return weights
        except (FileNotFoundError, OSError) as e:
            print(f"Error loading model at {model_path}: {e}. Waiting and retrying...")
            time.sleep(10)

def get_cohorts(NUM_CLIENTS, drifted_clients = None, drift = False):

    """
    Get cohorts of clients based on the weights of their loaded Keras models.

    Parameters:
    - NUM_CLIENTS (int): Number of clients participating in the federated learning.
    - drifted_clients (list): List of client IDs for clients that have drifted.
    - drift (bool): Flag indicating whether drift is considered.

    Returns:
    - dict: Dictionary representing cohorts with cohort number as keys and list of
            client IDs as values.
    """

    weights_list = []

    if drift:
        if drifted_clients is None:
            raise ValueError("List of drifted clients must be provided when drift=True")

        for client_id in drifted_clients:
            model_path = os.path.join(BASE_PATH, f"Results", f"client{client_id}", f"Models", f"client{client_id}_retrained_model.h5")
            weights_list.append(load_client_model(model_path))
        
        # Use the actual client IDs of drifted clients
        client_ids = drifted_clients
    else:
        for i in tqdm(range(1, NUM_CLIENTS + 1)):
            model_path = os.path.join(BASE_PATH, f"Results", f"client{i}", "Models", "base_model.h5")
            weights_list.append(load_client_model(model_path))
        
        # Use sequential client IDs if drift is not considered
        client_ids = list(range(1, NUM_CLIENTS + 1))

    weights = np.array(weights_list)

    # PCA transformation
    pc = PCA(n_components=0.99, svd_solver='full')
    W_transformed = pc.fit_transform(weights)

    # Find optimal NUM_COHORTS using silhouette score
    best_num_cohorts = find_optimal_cohorts(W_transformed)

    # Use the best NUM_COHORTS
    kmeans = KMeans(n_clusters=best_num_cohorts, init='k-means++')
    kmeans.fit(W_transformed)
    cohorts = kmeans.labels_.tolist()  # Convert to list
    cohorts = create_cohorts(best_num_cohorts, cohorts, client_ids)
    return cohorts

def get_cohorts_after_drift(NUM_CLIENTS, current_cluster, drifted_clients):
    """
    Adjust client cohorts after a drift event.

    Parameters:
    - NUM_CLIENTS (int): Number of clients participating in the federated learning.
    - current_cluster (dict): Current client cohorts represented as a dictionary,
                             with cohort number as keys and a list of client IDs as values.
    - drifted_clients (list): List of client IDs for clients that have drifted.

    Returns:
    - dict: Updated client cohorts after accounting for drift.
            The dictionary has cohort numbers as keys and lists of client IDs as values.
    Notes:
    - This function handles the adjustment of client cohorts in the event of drift.
    - If only one client has drifted, it splits the cohort containing the drifted client.
    - If multiple clients have drifted, it invokes the get_cohorts function with drift=True
      to obtain new cohorts based on the drifted clients' model weights.
    - If no clients have drifted, it returns the current cohort without any modifications.
    - The get_cohorts function used with drift=True internally updates the cohorts based on
      the weighted models of the drifted clients, ensuring that the clustering is adapted
      to the changes introduced by the drift.

    Example:
    ```
    NUM_CLIENTS = 10
    current_cluster = {1: [1, 2, 3], 2: [4, 5, 6], 3: [7, 8, 9, 10]}
    drifted_clients = [3, 6]

    updated_cohorts = get_cohorts_after_drift(NUM_CLIENTS, current_cluster, drifted_clients)
    ```

    In this example, if clients 3 and 6 have drifted, the function adapts the cohorts to
    reflect the changes introduced by the drift, providing an updated_cohorts dictionary.
    """
    if len(drifted_clients) == 1:
        # If only one client has drifted, split the cluster containing the drifted client
        new_cluster = split_clusters(current_cluster, drifted_clients)
        return new_cluster

    elif len(drifted_clients) > 1:
        # If multiple clients have drifted, obtain new cohorts based on their weights
        new_cluster = get_cohorts(NUM_CLIENTS, drifted_clients, drift=True)
        return new_cluster

    else:
        # If no clients have drifted, return the current cluster without modifications
        return current_cluster

