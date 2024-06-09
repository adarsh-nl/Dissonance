import numpy as np
import pandas as pd
import json
import re
import random

from Custom_Scripts.debug import debug
from Custom_Scripts.constants import BASE_PATH

def introduce_concept_drift(seed=None):
    """
    Introduce concept drift to a DataFrame.

    Args:
    - seed (int or None): Seed value for random number generation.

    Returns:
    - pd.DataFrame: DataFrame with introduced concept drift.
    """
    # Set the seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)

    original_data = pd.read_csv(BASE_PATH + 'Data\pronto.csv')

    # Sample 1000 samples for each class
    sampled_data = original_data.groupby('Fault').apply(lambda x: x.sample(n=1000, replace=True)).reset_index(drop=True)

    # randomly select two labels and shuffle them independently
    unique_labels = np.unique(sampled_data['Fault'])
    labels_to_shuffle = random.sample(list(unique_labels), 3)
    print(labels_to_shuffle)
    sampled_data['Fault'] = sampled_data['Fault'].apply(lambda x: random.choice(labels_to_shuffle) if x in labels_to_shuffle else x)
    sampled_data['Fault'] = np.random.permutation(sampled_data['Fault'])  # Shuffle the rest of the labels

    return sampled_data

def introduce_covariate_drift():

    """
    Introduce covariate drift to a DataFrame.

    Returns:
    - pd.DataFrame: DataFrame with introduced covariate drift.
    """

    # Load CSV file into a DataFrame
    df = pd.read_csv(BASE_PATH + 'Data\pronto.csv')

    #Sample 1000 samples for each class
    sampled_data = df.groupby('Fault').apply(lambda x: x.sample(n=250, replace=True)).reset_index(drop=True)

    # Extract labels from the last column
    labels = sampled_data.iloc[:, -1]

    # Generate a random array with the shape of the DataFrame's columns (excluding the last one)
    random_array = np.random.rand(*sampled_data.iloc[:, :-1].shape)

    # Replace the original values in the DataFrame with the random array
    sampled_data.iloc[:, :-1] = random_array

    # Add the labels back to the DataFrame
    sampled_data['Fault'] = labels

    # Loop through each column and introduce drift
    for column in sampled_data.columns[:-1]:
        # Randomly select parameters for the new distribution (e.g., Laplace distribution)
        new_loc = sampled_data[column].mean() + np.random.randint(1000) #randomly increase the mean of the column
        new_scale = np.random.uniform(0, (sampled_data[column].max() - sampled_data[column].min()) / 4)

        # Generate a random distribution for the modified feature
        new_distribution = np.random.laplace(loc=new_loc, scale=new_scale, size=len(sampled_data))

        # Replace the original values of the specified feature with the new distribution
        sampled_data[column] = new_distribution


    return sampled_data

def find_clients(string):
    return re.findall(r'\d+', string)

def drift(data, client_id, FL_comm_round, seed = 10):

    """
    Get drift data for a specific communication round and client.

    Parameters:
    - data (dict): Dictionary containing previous drift information.
    - client_id (int): ID of the client for which drift information is requested.
    - FL_comm_round (int): Communication round for which drift information is requested.

    Returns:
    - dataframe: Drifted data for the specified communication round and client.
    """

    file_path = BASE_PATH + "Results\Drift\Json\drift_entry.json"

    try:
        with open(file_path, 'r') as file:
            json_data = json.load(file)
            
        communication_round = find_clients(json_data.get('communication_round'))
        communication_round = list(map(int, communication_round))
        default_client_id = find_clients(json_data.get('client_id'))
        default_client_id = list(map(int, default_client_id))
        drift_type = json_data.get('drift_type')

        debug(f"Function for giving drift data in communication round {FL_comm_round+1} is started.")

        if FL_comm_round in communication_round:
            if client_id in default_client_id:
                if drift_type == "Concept Drift":
                    debug("Concept drift data incoming.")
                    return introduce_concept_drift(seed=seed)
                elif drift_type == "Covariate Drift":
                    debug("Covariate drift data incoming")
                    return introduce_covariate_drift
            
        # Default case or no drift condition
        return data
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return data
