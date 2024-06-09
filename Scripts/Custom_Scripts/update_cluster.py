from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate

from Custom_Scripts.constants import DISTANCE_THRESHOLD

def split_clusters(current_cluster, drifted_clients):

    """
    Split clusters based on drifted clients and create new clusters for drifted clients.

    Args:
    - current_cluster (dict): Dictionary representing the current clusters with cluster IDs as keys and lists of client IDs as values.
    - drifted_clients (list): List of client IDs that have drifted.

    Returns:
    - dict: A new dictionary representing updated clusters after considering drifted clients.
    """

    new_cluster = {}

    for key, value in current_cluster.items():
        new_values = [client_id for client_id in value if client_id not in drifted_clients]

        # If there are clients remaining in the cluster, update the original cluster
        if new_values:
            new_cluster[key] = new_values

    # Create new clusters for drifted clients
    new_cluster_count = max(current_cluster.keys(), default=0) + 1  # Start with a new count

    for client_id in drifted_clients:
        # Check if the client_id is already in any cluster
        in_existing_cluster = any(client_id in value for value in new_cluster.values())

        if not in_existing_cluster:
            # If not, add a new cluster for the drifted client with a unique key
            while new_cluster_count in new_cluster:
                new_cluster_count += 1  # Ensure unique key
            new_cluster[new_cluster_count] = [client_id]

    # Check if all clients are included in the final clusters
    all_clients = [client_id for value in new_cluster.values() for client_id in value]
    missing_clients = set(drifted_clients) - set(all_clients)

    # Add missing clients to the appropriate cluster
    for client_id in missing_clients:
        while new_cluster_count in new_cluster:
            new_cluster_count += 1  # Ensure unique key
        new_cluster[new_cluster_count] = [client_id]

    return new_cluster

def merge_models(model1, model2):

    """
    Merge two Keras models by concatenating their outputs.

    Args:
    - model1 (tf.keras.Model): First Keras model to be merged.
    - model2 (tf.keras.Model): Second Keras model to be merged.

    Returns:
    - tf.keras.Model: Merged Keras model with concatenated outputs from both input models.

    Raises:
    - ValueError: If the input and output shapes of the models are not the same, or if the architectures do not match.
    """

    # Ensure that the architectures of both models are the same
    if model1.input_shape != model2.input_shape or model1.output_shape != model2.output_shape:
        raise ValueError("Input and output shapes of the models must be the same.")

    # Create a list to hold the outputs of both models
    merged_outputs = []

    # Iterate through layers of both models and concatenate their outputs
    for layer1, layer2 in zip(model1.layers, model2.layers):
        if layer1.get_config() == layer2.get_config():  # Check if layers have the same configuration
            merged_outputs.append(concatenate([layer1.output, layer2.output]))
        else:
            raise ValueError("Architectures of the models do not match.")

    # Create the merged model
    merged_model = Model(inputs=model1.input, outputs=merged_outputs)

    return merged_model

def merge_cluster(cluster, num1, num2):

    """
    Merge clusters in a dictionary.

    Args:
    - cluster (dict): Dictionary representing clusters with keys as cluster numbers and values as lists of numbers in each cluster.
    - num1 (int): First number to be merged.
    - num2 (int): Second number to be merged.

    Returns:
    - dict: Updated dictionary representing clusters after merging.

    Note:
    The function merges clusters based on the input numbers. If the numbers are already in the same cluster, the original dictionary is returned.
    If the numbers are in different clusters, it merges the clusters containing each number or creates a new cluster if neither number is found.

    Example:
    >>> merge_cluster({1: [1, 2, 3], 2: [4, 5]}, 2, 4)
    {1: [1, 2, 3, 4, 5]}
    """

    # Check if the numbers are already in the same cluster
    for key, values in cluster.items():
        if num1 in values and num2 in values:
            return cluster

    # Check if the numbers are in different clusters
    key_num1, key_num2 = None, None
    for key, values in cluster.items():
        if num1 in values:
            key_num1 = key
        if num2 in values:
            key_num2 = key

    # If both numbers are in different clusters, merge them
    if key_num1 is not None and key_num2 is not None:
        cluster[key_num1] = list(set(cluster[key_num1] + cluster[key_num2]))
        del cluster[key_num2]
    # If only one number is found, add the other to its cluster
    elif key_num1 is not None:
        cluster[key_num1].append(num2)
    elif key_num2 is not None:
        cluster[key_num2].append(num1)
    # If neither number is found, create a new cluster
    else:
        cluster[len(cluster) + 1] = [num1, num2]

    return cluster

def put_clients_together(clients):

    """
    Put clients together into groups.

    Args:
    - clients (list): List of sets, where each set represents a client.

    Returns:
    - list: List of tuples, where each tuple represents a group of clients.

    Note:
    The function groups clients based on common elements (clients that have at least one element in common). 
    It iterates through the list of clients and forms groups by merging sets that have common elements.

    Example:
    >>> put_clients_together([{1, 2, 3}, {3, 4, 5}, {6, 7}])
    [(1, 2, 3, 4, 5), (6, 7)]
    """

    result = []

    for client in clients:
        added = False
        for group in result:
            if any(c in group for c in client):
                group.update(client)
                added = True
                break

        if not added:
            result.append(set(client))

    return [tuple(group) for group in result]

def find_similar_clients(drifted_clients, Distance_metric, current_cluster):

    """
    Find and cluster similar clients based on a distance metric.

    Args:
    - drifted_clients (list): List of client IDs that have drifted.
    - Distance_metric (numpy.ndarray): Distance metric matrix between clients.
    - current_cluster (dict): Dictionary representing the current clustering of clients.

    Returns:
    - tuple: A tuple containing updated current_cluster and already_clustered.

    Note:
    The function checks for pairs of drifted clients (i, j) that have a distance below DISTANCE_THRESHOLD
    based on the given Distance_metric. It then merges these clients into the same cluster in the current_cluster dictionary.

    Example:
    >>> find_similar_clients([1, 2, 3], distance_matrix, {1: [1], 2: [2], 3: [3]})
    ({1: [1, 2, 3]}, [(1, 2), (1, 3), (2, 3)])
    """

    not_similar = []
    already_clustered = []

    for i in drifted_clients:
        for j in drifted_clients:
            if i == j:
                pass
            elif(((i,j) not in already_clustered) and ((j,i) not in already_clustered) ):
                if (Distance_metric[i-1,j-1]) < DISTANCE_THRESHOLD:
                    #call the merge cluster function
                    current_cluster = merge_cluster(current_cluster, i , j)
                    
                    #to avoid confusion, the add the merged clients to the updated clusters.
                    already_clustered.append((i,j))
                    already_clustered.append((j,i))
                else:
                    not_similar.append((i,j))
                    not_similar.append((j,i))

    already_clustered = put_clients_together(already_clustered)
    return current_cluster, already_clustered
