import sys
sys.path.append("../")

from Custom_Scripts.constants import BASE_PATH
from Custom_Scripts.server_subModules import load_previous_model
from Custom_Scripts.model_architecture import model_initialisation
from Custom_Scripts.create_cohorts import get_cohorts_after_drift

import os
import time
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.stats import ks_2samp
from tensorflow.keras.models import load_model


def ks_2samp_test_and_plot(data_ref, data1, title, plot_path, separate_plots=False, transparent=True):
    """
    Perform two-sample Kolmogorov-Smirnov test and plot the results.

    Args:
    - data_ref (array-like): Reference data for the test.
    - data1 (array-like): Data to be compared with the reference data.
    - title (str): Title for the plot.
    - separate_plots (bool): Whether to plot histograms and CDFs separately (default is True).
    - transparent (bool): Whether to make the plot background transparent (default is False).

    Returns:
    - result (KS2sampResult): Result of the two-sample Kolmogorov-Smirnov test.

    Description:
    This function performs a two-sample Kolmogorov-Smirnov test to compare the distributions of two datasets.
    It calculates and plots histograms and cumulative distribution functions (CDFs) for the reference data and the data to be tested.
    The test results are printed, indicating whether the two datasets likely come from the same distribution.

    Steps:
    1. Sort the reference and test data for cumulative distribution function (CDF) calculation.
    2. Calculate the CDFs for both datasets.
    3. Perform the two-sample Kolmogorov-Smirnov test.
    4. Print the test results including the Kolmogorov-Smirnov statistic (D) and p-value.
    5. Plot histograms and CDFs of the reference and test data.
    6. Optionally, save the plot to a file.

    Note:
    - The two-sample Kolmogorov-Smirnov test is a non-parametric hypothesis test for testing whether two samples are drawn
      from the same continuous distribution.
    - Separate plots are created by default, but the user can specify to plot histograms and CDFs together.
    - Transparency can be enabled to create plots with transparent backgrounds.
    """

    # Sort data for CDF calculation
    sorted_data_ref = np.sort(data_ref)
    sorted_data1 = np.sort(data1)
    
    # Calculate unique sorted values and their corresponding CDFs
    unique_data_ref, counts_ref = np.unique(sorted_data_ref, return_counts=True)
    cdf_ref = np.cumsum(counts_ref) / len(data_ref)
    
    unique_data1, counts1 = np.unique(sorted_data1, return_counts=True)
    cdf1 = np.cumsum(counts1) / len(data1)
    
    # Perform the two-sample Kolmogorov-Smirnov test
    result = ks_2samp(data_ref, data1)

    # Print the test results
    print(f"Kolmogorov-Smirnov statistic (D): {result.statistic:.4f} ; p-value: {result.pvalue:.4f}")
    if result.pvalue > 0.05:
        print("The two datasets likely come from the same distribution.")
    else:
        print("The two datasets may come from different distributions.")
    
    # Plot the results
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot PDFs (histograms)
    alpha = 0.7 if transparent else 1.0
    axs[0].hist(sorted_data_ref, bins=30, density=True, label='Server Weights', alpha=alpha, color='b', linestyle='--')
    axs[0].hist(sorted_data1, bins=30, density=True, label='Client Weights', alpha=alpha, color='g')
    axs[0].set_ylabel('Density')
    axs[0].set_xlabel('Value')
    axs[0].legend()
    axs[0].set_title('PDF Comparison')

    # Plot CDFs
    axs[1].plot(unique_data_ref, cdf_ref, label='Server Weights', color='b', linestyle='--')
    axs[1].plot(unique_data1, cdf1, label='Client Weights', color='g')
    axs[1].set_xlabel('Value')
    axs[1].set_ylabel('CDF')
    axs[1].legend()
    axs[1].set_title('CDF Comparison')

    plt.suptitle(title)
    plt.savefig(plot_path)
    return result

def check_if_path_exists(path):

    """
    Check if the specified path exists. If it doesn't, create the directory structure.

    Parameters:
    - path (str): The path to be checked.

    Returns:
    - None: If the path already exists.
    
    Description:
    This function checks if the specified path exists. If the path does not exist, it creates the directory structure. 
    If the path already exists, the function returns without taking any action.

    Example:
    check_if_path_exists('/home/user/new_directory')
    # If the directory '/home/user/new_directory' does not exist, it will be created.
    # If the directory '/home/user/new_directory' already exists, the function will return without taking any action.

    Note:
    - This function is useful for ensuring that a specific directory exists before performing operations like file I/O.
    """

    if not os.path.exists(path):
        os.makedirs(path)
    else:
        return

def get_weights(loaded_model):

    """
    Retrieve the weights of each layer from a loaded neural network model.

    Parameters:
    - loaded_model (object): A loaded neural network model object, typically obtained after loading a model from disk.

    Returns:
    - model_weights_flat (ndarray): Flattened array containing the weights of all layers concatenated.

    Description:
    This function iterates through each layer of the provided loaded_model and retrieves the weights associated with each layer. 
    It then flattens these weights and concatenates them into a single 1-dimensional array.

    Example:
    # Assuming 'loaded_model' is a Keras model loaded from disk
    weights = get_weights(loaded_model)
    print(weights)
    # Output: Flattened array containing the concatenated weights of all layers in the model.

    Note:
    - The 'loaded_model' parameter should be a model object obtained after loading a model from disk, for example, using functions like Keras' `load_model`.
    - This function is particularly useful when you need to inspect or manipulate the weights of a pre-trained neural network model.
    """

    # Get the weights of each layer
    model_weights = []
    for layer in loaded_model.layers:
        layer_weights = layer.get_weights()
        model_weights.extend([w.flatten() for w in layer_weights])

    # Concatenate all flattened weights
    model_weights_flat = np.concatenate(model_weights)
    return model_weights_flat

def find_cohortID_with_clientID(cluster, client_ID):

    """
    Find the cohort ID associated with a given client ID within a cluster configuration.

    Parameters:
    - cluster (dict): A dictionary representing the cluster configuration, where keys are cohort IDs and values are lists of client IDs.
    - client_ID (int): The client ID to search for within the cluster.

    Returns:
    - cohortID (int or None): The cohort ID associated with the specified client ID if found, otherwise None.

    Description:
    This function iterates through each cohort ID and its associated list of client IDs in the provided cluster configuration.
    It checks if the specified client ID exists in any of the lists of client IDs. If found, it returns the cohort ID associated with that client ID.
    If the client ID is not found in any list, the function returns None.

    Example:
    cluster = {0: [101, 102, 103], 1: [104, 105], 2: [106, 107, 108]}
    client_id = 105
    cohort_id = find_cohortID_with_clientID(cluster, client_id)
    print(cohort_id)
    # Output: 1 (since client_id 105 is found in cohort ID 1)

    Note:
    - This function assumes that client IDs are integers.
    - It is useful for finding the cohort ID associated with a specific client within a cluster configuration.
    """

    for cohortID, lst in cluster.items():
        if client_ID in lst:
            return cohortID
    return None  # Return None if the integer is not found in any list

def load_model_weights(model_path):
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
            print(f'model path: {model_path}')
            model = load_model(model_path)
            weights = get_weights(model)
            return weights
        except (FileNotFoundError, OSError) as e:
            print(f"Error loading model at {model_path}: {e}. Waiting and retrying...")
            time.sleep(10)

def ks_plot_between_server_and_clients(server_weights, clients_weights, drifted_clients, title, plot_path):
    # Plot the results
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Plot PDFs (histograms)
    alpha = 0.5
    axs[0].hist(server_weights, bins=30, density=True, label='Server Weights', alpha=alpha, color='b', linestyle='--')
    
    # Get a colormap
    colormap = plt.cm.get_cmap('tab10')

    for i, client_weights in enumerate(clients_weights):
        color = colormap(i % 10)  # Use modulo to wrap around the colormap if needed
        axs[0].hist(client_weights, bins=30, density=True, label=f'Client {drifted_clients[i]} Weights', alpha=alpha, color=color)
    
    axs[0].set_ylabel('Density')
    axs[0].set_xlabel('Value')
    axs[0].legend()
    axs[0].set_title('PDF Comparison')

    # Plot CDFs
    server_weights_sorted = np.sort(server_weights)
    clients_cdfs = [np.searchsorted(np.sort(client_weights), server_weights_sorted, side='right') / len(client_weights) for client_weights in clients_weights]

    axs[1].plot(server_weights_sorted, np.arange(1, len(server_weights_sorted) + 1) / len(server_weights_sorted), label='Server Weights', color='b', linestyle='--')
    for i, client_cdf in enumerate(clients_cdfs):
        color = colormap(i % 10)  # Use modulo to wrap around the colormap if needed
        axs[1].plot(server_weights_sorted, client_cdf, label=f'Client {drifted_clients[i]} Weights', color=color)
    
    axs[1].set_xlabel('Value')
    axs[1].set_ylabel('CDF')
    axs[1].legend()
    axs[1].set_title('CDF Comparison')

    plt.suptitle(title)
    plt.savefig(plot_path)
    plt.show()  # Show the plot

def load_and_get_weights(client_id):
    return load_model_weights(BASE_PATH + f"Results\client{client_id}\Models\client{client_id}_retrained_model.h5")

def update_clusters(old_cluster, new_cluster):

    """
    Update the existing cluster configuration with new clusters and their members.

    Parameters:
    - old_cluster (dict): A dictionary representing the existing cluster configuration, where keys are cluster IDs and values are lists of member IDs.
    - new_cluster (dict): A dictionary representing the new cluster configuration, where keys are cluster IDs and values are lists of member IDs.

    Returns:
    - updated_cluster (dict): A dictionary representing the updated cluster configuration after incorporating new clusters. Keys are cluster IDs and values are lists of member IDs.

    Algorithm:
    1. Initialize an empty dictionary called 'updated_cluster' to hold the updated cluster configuration.
    2. Find the next available cluster ID by incrementing the maximum cluster ID in the old_cluster by 1.
    3. Process old clusters:
       a. Iterate through each cluster ID and its members in the old_cluster.
       b. Check if any members of the current cluster are not present in any of the new clusters. If so, add these members to the updated_cluster with the same cluster ID.
    4. Process new clusters:
       a. Iterate through each cluster ID and its members in the new_cluster.
       b. If the cluster ID already exists in the updated_cluster, assign a new cluster ID for the current cluster and increment next_cluster_id.
       c. Add the cluster ID and its members to the updated_cluster.

    Example:
    # Example usage
    old_cluster = {1: [1, 2, 3, 4], 2:[5, 6, 7, 8, 9, 10]}
    new_cluster = {1: [1,2], 2:[3,4], 3:[10]}

    output: {2: [5, 7, 8, 9], 1: [1, 2], 3: [3, 4], 4: [10], 5: [6]}

    Note:
    - This function assumes that cluster IDs are integers and are consecutive starting from 0.
    - If the old_cluster is empty, the function assumes the next available cluster ID to start from 0.
    """

    updated_cluster = {}
    next_cluster_id = max(old_cluster.keys()) + 1  # ID for new clusters

    # Process old clusters
    for cluster_id, members in old_cluster.items():
        remaining_members = [m for m in members if m not in sum(new_cluster.values(), [])]  # Check all new clusters
        if remaining_members:
            updated_cluster[cluster_id] = remaining_members

    # Process new clusters
    for cluster_id, members in new_cluster.items():
        if cluster_id in updated_cluster:
            # If cluster ID already exists, create a new cluster with a new ID
            updated_cluster[next_cluster_id] = members
            next_cluster_id += 1
        else:
            updated_cluster[cluster_id] = members

    return updated_cluster

def get_kstest_result(client_id, comm_round, previous_cluster):

    """
    Get the result of the Kolmogorov-Smirnov (KS) test comparing the weights of a client's model with the server's model.

    Parameters:
    - client_id (int): The ID of the client whose model is being compared with the server's model.
    - comm_round (int): The communication round number.
    - previous_cluster (dict): A dictionary representing the previous cluster configuration, where keys are cohort IDs and values are lists of client IDs.

    Returns:
    - kstest_result.statistic (float): The test statistic of the KS test comparing the weights of the client's model with the server's model.

    Description:
    This function performs the following steps:
    1. Using the provided 'client_id' and 'previous_cluster', it finds the cohort ID associated with the client ID using the 'find_cohortID_with_clientID' function.
    2. It loads the server's model from the previous communication round (if available) using the 'load_previous_model' function. If the client is not associated with any cohort, it initializes a new server model using the 'model_initialisation' function.
    3. It loads the weights of both the server model and the client's retrained model.
    4. It performs a KS test (Kolmogorov-Smirnov test) comparing the distributions of weights between the server model and the client's model using the 'ks_2samp_test_and_plot' function.
    5. It generates a plot illustrating the KS test result and saves it to a specified plot path.
    6. It returns the test statistic of the KS test.

    Example:
    client_id = 123
    comm_round = 5
    previous_cluster = {0: [101, 102, 103], 1: [104, 105], 2: [106, 107, 108]}
    ks_statistic = get_kstest_result(client_id, comm_round, previous_cluster)
    print(ks_statistic)
    # Output: The test statistic of the KS test comparing the weights of the client's model with the server's model.

    Note:
    - This function assumes the existence of functions like 'find_cohortID_with_clientID', 'load_previous_model', 'model_initialisation', 'load_model_weights', and 'ks_2samp_test_and_plot'.
    - It is useful for evaluating the discrepancy between the weights of a client's model and the server's model in a federated learning setting.
    """

    cohortID = find_cohortID_with_clientID(previous_cluster, client_id)
    server_model = load_previous_model(comm_round-1, cohortID) if cohortID is not None else model_initialisation()
    server_model_weights = get_weights(server_model)
    client_model_weights = load_model_weights(BASE_PATH + f"Results\client{client_id}\Models\client{client_id}_retrained_model.h5")
    check_if_path_exists(BASE_PATH + f"Results\\server\\Plots\\Dissonance\\")
    kstest_result = ks_2samp_test_and_plot(server_model_weights, client_model_weights, f"KS TEST with server and client{client_id}", plot_path= BASE_PATH + f"Results\\server\\Plots\\Dissonance\\KSTest_result_{client_id}.png")
    return kstest_result.statistic

def fix_drift_and_cohort_clients_dissonance(NUM_CLIENTS, comm_round, previous_cluster, current_cluster, drifted_clients):

    """
    Update the cluster configuration to fix drift and dissonance among cohort clients.

    Parameters:
    - NUM_CLIENTS (int): Total number of clients in the system.
    - comm_round (int): Current communication round.
    - previous_cluster (dict): Cluster configuration from the previous round, where keys are cohort IDs and values are lists of client IDs.
    - current_cluster (dict): Current cluster configuration, where keys are cohort IDs and values are lists of client IDs.
    - drifted_clients (list): List of client IDs that have drifted from the previous round.

    Returns:
    - updated_cluster (dict): Updated cluster configuration after fixing drift and dissonance.

    Description:
    This function addresses drift and dissonance issues among cohort clients by updating the cluster configuration.
    It identifies clients with drifted behavior using a Kolmogorov-Smirnov (KS) test comparing their model weights with the server's.
    Clients with model weights significantly different from the server's are considered drifted.
    For each drifted client, it determines if the KS test result is above or below a reference threshold (usually 0).
    Based on the test result, it divides drifted clients into two groups: above_reference and below_reference.
    For each group, it creates new cohorts using the drifted clients and updates the cluster configuration accordingly.
    The updated cluster configuration is then returned.

    Example:
    NUM_CLIENTS = 10
    comm_round = 3
    previous_cluster = {0: [1, 2, 3], 1: [4, 5]}
    current_cluster = {0: [1, 2, 3], 1: [4, 5], 2: [6, 7]}
    drifted_clients = [6, 7, 8]
    updated_cluster = fix_drift_and_cohort_clients_dissonance(NUM_CLIENTS, comm_round, previous_cluster, current_cluster, drifted_clients)
    print(updated_cluster)
    # Output: Updated cluster configuration after fixing drift and dissonance.

    Note:
    - This function assumes that the `get_kstest_result`, `get_cohorts_after_drift`, and `update_clusters` functions are implemented elsewhere in the codebase.
    - It is designed to be used within a federated learning framework to handle changes in client behavior over communication rounds.
    """

    above_reference = [client_id for client_id in drifted_clients if (kstest_result := get_kstest_result(client_id, comm_round, previous_cluster)) > 0]
    below_reference = [client_id for client_id in drifted_clients if (kstest_result := get_kstest_result(client_id, comm_round, previous_cluster)) <= 0]
    if len(above_reference)>0:
        new_cluster = get_cohorts_after_drift(NUM_CLIENTS, current_cluster, drifted_clients = above_reference)
        updated_cluster = update_clusters(current_cluster, new_cluster)
    if len(below_reference)>0:
        new_cluster  = get_cohorts_after_drift(NUM_CLIENTS, current_cluster, drifted_clients = below_reference)
        updated_cluster = update_clusters(updated_cluster, new_cluster)

    # # Using list comprehension
    # cohortID = find_cohortID_with_clientID(previous_cluster, random.choice(np.unique(drifted_clients)))
    # server_model = load_previous_model(comm_round-1, cohortID) if cohortID is not None else model_initialisation()
    # server_model_weights = get_weights(server_model)
    # drifted_model_weights = [load_and_get_weights(client_id) for client_id in drifted_clients]
    # ks_plot_between_server_and_clients(server_model_weights, drifted_model_weights, drifted_clients, "PDF and CDF of server and drifted clients", BASE_PATH + f"Results\server\Plots\Dissonance\KSTest_combined_plot.png")

    return updated_cluster