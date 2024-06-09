#Gives access to all the scripts in the directory
import sys
sys.path.append('../')

#boiler plate code for all clients
from Custom_Scripts.debug import debug
from Custom_Scripts.model_architecture import model_initialisation
from Custom_Scripts.ClassPronto import Pronto
from Custom_Scripts.TrainTestSplit import custom_train_test_split
from Custom_Scripts.ClientID import create_clientid
from Custom_Scripts.ClassClients import Client_fn, load_previous_client_model, signal_status_completion, save_client_results, create_results_file
from Custom_Scripts.constants import BASE_PATH
from Custom_Scripts.server_subModules import update_status_file_as_not_completed, update_status_file_as_completed, save_cluster_to_file, wait_for_clients_completion, save_trained_server_model, load_previous_model, get_ready_clients, server_functions, create_neccessary_text_files
from Custom_Scripts.merge_models import save_new_cohort_model, merge_models
from Custom_Scripts.update_cluster import find_similar_clients
from Custom_Scripts.server_subModules import save_trained_server_model, save_results, fit_config, get_eval_fn, save_cluster_to_file, evaluate_model_and_get_metrics, save_metrics
from Custom_Scripts.constants import BASE_PATH
from Custom_Scripts.create_cohorts import get_cohorts, get_cohorts_after_drift
from Custom_Scripts.dissonance_subfunctions import fix_drift_and_cohort_clients_dissonance

#import necessary libraries
import os
import numpy as np
import pandas as pd
import warnings
from collections import deque
import matplotlib.pyplot as plt
import time
import flwr as fl
import streamlit as st
import ast

# Filter out Keras warnings about saving models in HDF5 format
warnings.filterwarnings("ignore", category=UserWarning, module=".*tensorflow.keras.*")

data = pd.read_csv(BASE_PATH + "Data\server.csv")
X_train, X_test, y_train, y_test = custom_train_test_split(data)
client_id = create_clientid(os.path.splitext(os.path.basename(__file__))[0])
client = Client_fn(client_id, X_train, X_test, y_train, y_test)

#Define constants
DEBUG = True

def main(X_train, X_val, y_train, y_val, model, comm_round, cohort, min_fit_clients, min_available_clients):

    """
    Main function for federated learning on the server side.

    Args:
    - X_train (ndarray): Features of the training data.
    - X_val (ndarray): Features of the validation data.
    - y_train (ndarray): Labels of the training data.
    - y_val (ndarray): Labels of the validation data.
    - model (keras.Model): Initial machine learning model for federated learning.
    - comm_round (int): Current communication round number.
    - cohort (str): Identifier for the current cohort.
    - min_fit_clients (int): Minimum number of clients required for fitting the model.
    - min_available_clients (int): Minimum number of clients available for training.

    Returns:
    - model (keras.Model): Trained machine learning model.

    Description:
    This function orchestrates the federated learning process on the server side. It initializes the federated learning
    strategy, starts the Flower server for federated learning, saves the metrics for analysis, and returns the trained
    model.

    Federated Learning Strategy:
    - Strategy: Federated Averaging (FedAvg)
    - Evaluation Function: Calculates evaluation metrics on the validation data.
    - Fit Configuration Function: Defines configuration for fitting the model.
    - Initial Parameters: Initial model weights converted to Flower parameters.
    - Fraction Fit: Fraction of clients to use for model fitting (1.0 for all clients).
    - Minimum Fit Clients: Minimum number of clients required for fitting the model.
    - Minimum Available Clients: Minimum number of clients available for training.

    Server Configuration:
    - Server Address: localhost:8083 (default Flower server address).
    - Number of Rounds: 1 (single round of federated learning). 
        *** Because, Flwr does not support recent FL algorithms, so we had to implement them manually by iterating them comm_round times***

    Steps:
    1. Initialize the federated learning strategy with the specified parameters.
    2. Start the Flower server for federated learning with the defined strategy and configuration.
    3. Save the centralized losses obtained during the federated learning process.
    4. Save the results including communication round, cohort, and federated learning history.
    5. Return the trained model.

    Note:
    - The function relies on Flower for federated learning orchestration and assumes Flower server is running at the
      specified address.
    """

    strategy = fl.server.strategy.FedAvg(
        evaluate_fn = get_eval_fn(model, X_val, y_val),
        on_fit_config_fn=fit_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
        fraction_fit = 1.0,
        min_fit_clients  = min_fit_clients,
        min_available_clients = min_available_clients,
    )    

    debug("SERVER>>>> Running now!")
    # Start Flower server for five rounds of federated learning
    history_ = fl.server.start_server(server_address="localhost:8083",  strategy=strategy, config=fl.server.ServerConfig(num_rounds=1))
    
    #save the metrics for plots and analysis
    debug("losses_ centralised", history_.losses_centralized)
    
    save_results(comm_round, cohort, history_)
    
    return model

def cdafedavg(FL_round, NUM_CLIENTS, current_cluster, Use_Model_Parameters_for_cohorting):

    """
    Federated Learning process with CDA-FedAvg (Conept Drift-Aware Federated Averaging) algorithm.

    Args:
    - FL_round (int): Total number of communication rounds for the federated learning process.
    - NUM_CLIENTS (int): Total number of clients participating in the federated learning process.
    - current_cluster (dict): Dictionary containing the current clustering of clients. Keys represent cohort identifiers
      and values represent lists of client IDs belonging to each cohort.
    - Use_Model_Parameters_for_cohorting (int): Flag indicating whether to use model parameters for cohorting (1 for True,
      0 for False).

    Returns:
    - None

    Description:
    This function implements the CDA-FedAvg algorithm for federated learning. It trains a deep learning model
    collaboratively across multiple clients while considering the clustered nature of the data. The process involves
    several communication rounds where clients perform local training on their data and communicate model updates to the
    server, which aggregates them to update the global model. In each round, model evaluation metrics are computed and
    stored for analysis.

    This function performs the following steps:
    1. If 'Use_Model_Parameters_for_cohorting' is set to 1, it updates the 'current_cluster' based on model parameters.
       Otherwise, it uses the provided 'current_cluster'.
    2. Reads server data from a CSV file located at a predefined path.
    3. Splits the data into training and validation sets.
    4. Iterates over communication rounds, performing the following steps in each round:
       a. Resets the status file for the current communication round.
       b. Checks the number of cohorts:
          - If there's only one cohort, it performs federated learning on that cohort's clients.
          - If there are multiple cohorts, it performs federated learning on each cohort's clients separately.
       c. Waits for clients to complete their training for the current communication round.
       d. Updates the status file after all clients have completed the round.
       e. Saves the current clustering to a file.
       f. Sleeps for 30 seconds before starting the next communication round.

    Note:
    - The function relies on various helper functions such as 'get_cohorts', 'custom_train_test_split', 'debug',
      'model_initialisation', 'load_previous_model', 'get_ready_clients', 'main', 'save_trained_server_model',
      'evaluate_model_and_get_metrics', 'save_metrics', 'wait_for_clients_completion', 'update_status_file_as_not_completed',
      'update_status_file_as_completed', 'save_cluster_to_file', and 'time.sleep'.
    """

    if Use_Model_Parameters_for_cohorting == 1:
        current_cluster = get_cohorts(NUM_CLIENTS)

    previous_accuracy, current_accuracy, previous_loss, current_loss = 0, 0, 0, 0
    debug("Reading server data...")
    server_data = pd.read_csv(BASE_PATH + r"Data\server.csv")

    X_train, X_val, y_train, y_val = custom_train_test_split(server_data)

    for comm_round in range(1, FL_round+1):
        # creates all the text files for status checking and updating drift in data.
        #create_neccessary_text_files(comm_round) 

        # Reset the communication round status file
        update_status_file_as_not_completed(comm_round)

        no_cohorts = len(current_cluster)
        debug(f'SERVER>>>> Current Cluster: {current_cluster} Number of cohorts: {no_cohorts}')
        if no_cohorts == 1:
            key, values = list(current_cluster.items())[0]
            debug(f"SERVER>>>> No of cohorts: {no_cohorts}, Performing FL on cohort : {key} with clients {values}")
            get_ready_clients(values)

            if comm_round == 1:
                model = model_initialisation()
            else:
                #load the previous communication round model
                model = load_previous_model(comm_round, key)
                debug(f"SERVER>>>> Communication round: {comm_round-1} - Model: server_model_round{comm_round-1}.h5")
            
            model = main(X_train, X_val, y_train, y_val, model, comm_round, key, min_fit_clients= NUM_CLIENTS, min_available_clients= NUM_CLIENTS) # Need to pass the parameters as the per the number of clients in the cohort to trigger the FL procress.
            save_trained_server_model(model, comm_round, key)
            current_loss, delta_loss, current_accuracy, delta_accuracy = evaluate_model_and_get_metrics(model, X_val, y_val, previous_accuracy, previous_loss)
            previous_accuracy, previous_loss = current_accuracy, current_loss
            save_metrics(current_loss, delta_loss, current_accuracy, delta_accuracy)
            
        #wait for the clients to complete the communication round.  
        wait_for_clients_completion(comm_round, NUM_CLIENTS)

        # Update the status file after all clients have completed
        update_status_file_as_completed(comm_round)
        
        # Save the current_cluster dictionary to a file
        save_cluster_to_file(current_cluster, comm_round)
        time.sleep(30)

def feddrift(FL_round, NUM_CLIENTS, current_cluster, Use_Model_Parameters_for_cohorting):

    """
    Federated Learning process with Federated Drift Detection.

    Args:
    - FL_round (int): Total number of communication rounds for the federated learning process.
    - NUM_CLIENTS (int): Total number of clients participating in the federated learning process.
    - current_cluster (dict): Dictionary containing the current clustering of clients. Keys represent cohort identifiers
      and values represent lists of client IDs belonging to each cohort.
    - Use_Model_Parameters_for_cohorting (int): Flag indicating whether to use model parameters for cohorting (1 for True,
      0 for False).

    Returns:
    - None

    Description:
    This function implements the Federated Drift Detection algorithm for federated learning. It trains a deep learning
    model collaboratively across multiple clients while detecting and handling data drift. The process involves several
    communication rounds where clients perform local training on their data and communicate model updates to the server.
    Data drift detection is performed during certain communication rounds, and similar clients experiencing drift are
    identified and clustered together. Their models are then merged to handle the drift effectively.

    This function performs the following steps:
    1. If 'Use_Model_Parameters_for_cohorting' is set to 1, it updates the 'current_cluster' based on model parameters.
       Otherwise, it uses the provided 'current_cluster'.
    2. Reads server data from a CSV file located at a predefined path.
    3. Splits the data into training and validation sets.
    4. Iterates over communication rounds, performing the following steps in each round:
       a. Initializes a machine learning model.
       b. Creates necessary text files for status checking and updating data drift.
       c. Resets the status file for the current communication round.
       d. Checks the number of cohorts:
          - If there's only one cohort, it performs federated learning on that cohort's clients.
          - If there are multiple cohorts, it performs federated learning on each cohort's clients separately.
       e. If the current communication round is greater than 1, performs drift detection:
          - Identifies drifted and non-drifted clients.
          - Clusters drifted clients based on similarity using distance metric.
          - Merges models of similar clients to handle drift.
       f. Waits for clients to complete their training for the current communication round.
       g. Updates the status file after all clients have completed the round.
       h. Saves the current clustering to a file.
       i. Sleeps for 30 seconds before starting the next communication round.

    Note:
    - The function relies on various helper functions such as 'get_cohorts', 'custom_train_test_split', 'debug',
      'model_initialisation', 'load_previous_model', 'get_ready_clients', 'main', 'save_trained_server_model',
      'server_functions', 'find_similar_clients', 'merge_models', 'save_new_cohort_model', 'wait_for_clients_completion',
      'update_status_file_as_not_completed', 'update_status_file_as_completed', 'save_cluster_to_file', and 'time.sleep'.
    """

    if Use_Model_Parameters_for_cohorting == 1:
        current_cluster = get_cohorts(NUM_CLIENTS)

    debug("Reading server data...")
    server_data = pd.read_csv(BASE_PATH + r"Data\server.csv")

    X_train, X_val, y_train, y_val = custom_train_test_split(server_data)

    for comm_round in range(1, FL_round+1):        
        debug(f"SERVER>>>>, Beginning of COMMUNICATION ROUND: {comm_round}, Current Cluster: {current_cluster}")
        model = model_initialisation()

        # creates all the text files for status checking and updating drift in data.
        create_neccessary_text_files(comm_round) 

        # Reset the communication round status file
        update_status_file_as_not_completed(comm_round)
            
        no_cohorts = len(current_cluster)
        if no_cohorts == 1:
            key, values = list(current_cluster.items())[0]
            debug(f"SERVER>>>> No of cohorts: {no_cohorts}, Performing FL on cohort : {key} with clients {values}")
            get_ready_clients(values)

            if comm_round == 1:
                model = model_initialisation()
            else:
                #load the previous communication round model
                model = load_previous_model(comm_round, key)
                debug(f"SERVER>>>> Communication round: {comm_round-1} - Model: server_model_round{comm_round-1}.h5")
            
            model = main(X_train, X_val, y_train, y_val, model, comm_round, key, min_fit_clients=NUM_CLIENTS, min_available_clients=NUM_CLIENTS) # Need to pass the parameters as the per the number of clients in the cohort to trigger the FL procress.
            save_trained_server_model(model, comm_round, key)

        elif no_cohorts > 1:
            for key, values in current_cluster.items():
                if comm_round == 1:
                    model = model_initialisation()
                elif comm_round>1:
                    model = load_previous_model(comm_round, key)
                debug(f"SERVER>>>> No of cohorts: {no_cohorts}, Performing FL on cohort : {key} with clients {values}")
                get_ready_clients(values)
                #model = load_cohort_model(key, comm_round)
                model = main(X_train, X_val, y_train, y_val, model, comm_round, key, min_fit_clients=len(values), min_available_clients=len(values)) # Need to pass the parameters as the per the number of clients in the cohort to trigger the FL procress.
                save_trained_server_model(model, comm_round, key)

        if comm_round > 1:
            drifted_clients, non_drifted_clients, current_cluster = server_functions(comm_round, current_cluster)
            time.sleep(30)
            if (len(drifted_clients) > 0)  and (len(non_drifted_clients)<NUM_CLIENTS):

                time.sleep(60) # wait till the clients calculate and proceed. 
                
                #Load the distance metric and cluster the clients if needed.
                distance_metric = np.load(BASE_PATH + f'Results\DistanceMetric\communication_round{comm_round}_distancemetric.npy')

                #call the find_similar_clients function.
                current_cluster, similar_clients = find_similar_clients(drifted_clients, distance_metric, current_cluster)

                debug(f"SERVER>>>> Communication round: {comm_round}, Similar Clients: {similar_clients}, Updated cluster after clustering the similar clients: {current_cluster}")
                if len(similar_clients) == 0:
                    debug("SERVER>>>> No clients are experiencing similar drift.")
                else:
                    models = merge_models(similar_clients, comm_round)
                    save_new_cohort_model(current_cluster, similar_clients, models, comm_round)
            else:
                pass

        #wait for the clients to complete the communication round.  
        wait_for_clients_completion(comm_round, NUM_CLIENTS)

        # Update the status file after all clients have completed
        update_status_file_as_completed(comm_round)

        # Save the current_cluster dictionary to a file
        save_cluster_to_file(current_cluster, comm_round)

        debug(f"SERVER>>>>, End of COMMUNICATION ROUND: {comm_round}, Current Cluster: {current_cluster}")

        #main(model, comm_round) #Start FL server
        time.sleep(30)

def dissonance(FL_round, NUM_CLIENTS, current_cluster, Use_Model_Parameters_for_cohorting):
    """
    Perform federated learning with the Dissonance algorithm on the server side.

    Args:
    - FL_round (int): Number of federated learning rounds.
    - NUM_CLIENTS (int): Total number of clients participating in federated learning.
    - current_cluster (dict): Current cluster configuration.
    - Use_Model_Parameters_for_cohorting (int): Flag indicating whether to use model parameters for cohorting.

    Returns:
    - None

    Description:
    This function executes federated learning with the Dissonance algorithm on the server side. It coordinates the
    communication rounds, performs model initialization, triggers federated learning on each cohort, handles drift detection,
    and updates the current cluster configuration.

    Steps:
    1. Initialize necessary components such as model and data.
    2. Loop through each communication round:
        - Determine the cohort(s) for federated learning based on the current cluster configuration.
        - Perform federated learning on each cohort:
            - Load the previous model or initialize a new one.
            - Get ready clients within the cohort.
            - Train the model using federated learning.
            - Save the trained model.
        - Handle drift detection and update the current cluster configuration if necessary.
        - Wait for clients to complete the communication round.
        - Update status files and save the current cluster configuration.
    3. Pause briefly before the next communication round.

    Note:
    - The function assumes the availability of various helper functions such as model initialization, file handling,
      drift detection, client coordination, and status updates.
    - Drift detection and retraining are crucial steps to ensure model adaptation to changing data distributions.
    - The Dissonance algorithm focuses on maintaining model performance despite data drifts and variations.
    """
    if Use_Model_Parameters_for_cohorting == 1:
        current_cluster = get_cohorts(NUM_CLIENTS)

    debug("SERVER>>>> Reading data")
    server_data = pd.read_csv(BASE_PATH + r"Data\server.csv")

    X_train, X_val, y_train, y_val = custom_train_test_split(server_data)

    for comm_round in range(1, FL_round+1):        
        debug(f"SERVER>>>>, Beginning of COMMUNICATION ROUND: {comm_round}, Current Cluster: {current_cluster}")
        model = model_initialisation()

        # creates all the text files for status checking and updating drift in data.
        create_neccessary_text_files(comm_round) 

        # Reset the communication round status file
        update_status_file_as_not_completed(comm_round)
            
        no_cohorts = len(current_cluster)
        if no_cohorts == 1:
            key, values = list(current_cluster.items())[0]
            debug(f"SERVER>>>> No of cohorts: {no_cohorts}, Performing FL on cohort : {key} with clients {values}")
            get_ready_clients(values)

            if comm_round == 1:
                model = model_initialisation()
            else:
                #load the previous communication round model
                model = load_previous_model(comm_round, key)
                if model is None:
                    model = model_initialisation()
                debug(f"SERVER>>>> Communication round: {comm_round-1} - Model: server_model_round{comm_round-1}.h5")
            
            model = main(X_train, X_val, y_train, y_val, model, comm_round, key, min_fit_clients=NUM_CLIENTS, min_available_clients=NUM_CLIENTS) # Need to pass the parameters as the per the number of clients in the cohort to trigger the FL procress.
            save_trained_server_model(model, comm_round, key)

        elif no_cohorts > 1:
            for key, values in current_cluster.items():
                if comm_round == 1:
                    model = model_initialisation()
                elif comm_round>1:
                    model = load_previous_model(comm_round, key)
                    if model is None:
                        model = model_initialisation()
                debug(f"SERVER>>>> No of cohorts: {no_cohorts}, Performing FL on cohort : {key} with clients {values}")
                get_ready_clients(values)
                #model = load_cohort_model(key, comm_round)
                model = main(X_train, X_val, y_train, y_val, model, comm_round, key, min_fit_clients=len(values), min_available_clients=len(values)) # Need to pass the parameters as the per the number of clients in the cohort to trigger the FL procress.
                save_trained_server_model(model, comm_round, key)

        if comm_round > 1:
            drifted_clients, non_drifted_clients, new_cluster = server_functions(comm_round, current_cluster)
            previous_cluster = current_cluster; current_cluster = new_cluster
            time.sleep(30)
            if (len(drifted_clients) > 0)  and (len(non_drifted_clients)<NUM_CLIENTS):
                time.sleep(60) # wait till the clients retrain their model
                current_cluster = fix_drift_and_cohort_clients_dissonance(NUM_CLIENTS, comm_round, previous_cluster, current_cluster, drifted_clients)
                #current_cluster = get_cohorts_after_drift(NUM_CLIENTS, current_cluster, drifted_clients)
            else:
                pass

        #wait for the clients to complete the communication round.  
        wait_for_clients_completion(comm_round, NUM_CLIENTS)

        # Update the status file after all clients have completed
        update_status_file_as_completed(comm_round)

        # Save the current_cluster dictionary to a file
        save_cluster_to_file(current_cluster, comm_round)

        debug(f"SERVER>>>>, End of COMMUNICATION ROUND: {comm_round}, Current Cluster: {current_cluster}")

        #main(model, comm_round) #Start FL server
        time.sleep(30)

def Vannila_FL(FL_round, NUM_CLIENTS, current_cluster, Use_Model_Parameters_for_cohorting):

    """
    Federated Learning process with vanilla implementation.

    Args:
    - FL_round (int): Total number of communication rounds for the federated learning process.
    - NUM_CLIENTS (int): Total number of clients participating in the federated learning process.
    - current_cluster (dict): Dictionary containing the current clustering of clients. Keys represent cohort identifiers
      and values represent lists of client IDs belonging to each cohort.
    - Use_Model_Parameters_for_cohorting (int): Flag indicating whether to use model parameters for cohorting (1 for True,
      0 for False).

    Returns:
    - None

    Description:
    This function orchestrates the federated learning process. It trains a deep learning model collaboratively across
    multiple clients while preserving data privacy. The process involves several communication rounds where clients
    perform local training on their data and communicate model updates to the server, which aggregates them to update
    the global model.

    This function performs the following steps:
    1. If 'Use_Model_Parameters_for_cohorting' is set to 1, it updates the 'current_cluster' based on model parameters.
       Otherwise, it uses the provided 'current_cluster'.
    2. Reads server data from a CSV file located at a predefined path.
    3. Splits the data into training and validation sets.
    4. Iterates over communication rounds, performing the following steps in each round:
       a. Initializes a machine learning model.
       b. Creates necessary text files for status checking and updating data drift.
       c. Resets the status file for the current communication round.
       d. Checks the number of cohorts:
          - If there's only one cohort, it performs federated learning on that cohort's clients.
          - If there are multiple cohorts, it performs federated learning on each cohort's clients separately.
       e. Waits for clients to complete their training for the current communication round.
       f. Updates the status file after all clients have completed the round.
       g. Saves the current clustering to a file.
       h. Sleeps for 30 seconds before starting the next communication round.

    Note:
    - The function relies on various helper functions such as 'get_cohorts', 'custom_train_test_split', 'debug',
      'model_initialisation', 'create_neccessary_text_files', 'update_status_file_as_not_completed',
      'get_ready_clients', 'load_previous_model', 'main', 'save_trained_server_model', 'wait_for_clients_completion',
      'update_status_file_as_completed', 'save_cluster_to_file', and 'time.sleep'.
    """

    if Use_Model_Parameters_for_cohorting == 1:
        current_cluster = get_cohorts(NUM_CLIENTS)

    debug("Reading server data...")
    server_data = pd.read_csv(BASE_PATH + r"Data\server.csv")

    X_train, X_val, y_train, y_val = custom_train_test_split(server_data)

    for comm_round in range(1, FL_round+1):        
        debug(f"SERVER>>>>, Beginning of COMMUNICATION ROUND: {comm_round}, Current Cluster: {current_cluster}")
        model = model_initialisation()

        # creates all the text files for status checking and updating drift in data.
        create_neccessary_text_files(comm_round) 

        # Reset the communication round status file
        update_status_file_as_not_completed(comm_round)
            
        no_cohorts = len(current_cluster)
        if no_cohorts == 1:
            key, values = list(current_cluster.items())[0]
            debug(f"SERVER>>>> No of cohorts: {no_cohorts}, Performing FL on cohort : {key} with clients {values}")
            get_ready_clients(values)

            if comm_round == 1:
                model = model_initialisation()
            else:
                #load the previous communication round model
                model = load_previous_model(comm_round, key)
                debug(f"SERVER>>>> Communication round: {comm_round-1} - Model: server_model_round{comm_round-1}.h5")
            
            model = main(X_train, X_val, y_train, y_val, model, comm_round, key, min_fit_clients=NUM_CLIENTS, min_available_clients=NUM_CLIENTS) # Need to pass the parameters as the per the number of clients in the cohort to trigger the FL procress.
            save_trained_server_model(model, comm_round, key)

        elif no_cohorts > 1:
            for key, values in current_cluster.items():
                if comm_round == 1:
                    model = model_initialisation()
                elif comm_round>1:
                    model = load_previous_model(comm_round, key)
                debug(f"SERVER>>>> No of cohorts: {no_cohorts}, Performing FL on cohort : {key} with clients {values}")
                get_ready_clients(values)
                #model = load_cohort_model(key, comm_round)
                model = main(X_train, X_val, y_train, y_val, model, comm_round, key, min_fit_clients=len(values), min_available_clients=len(values)) # Need to pass the parameters as the per the number of clients in the cohort to trigger the FL procress.
                save_trained_server_model(model, comm_round, key)

        #wait for the clients to complete the communication round.  
        wait_for_clients_completion(comm_round, NUM_CLIENTS)

        # Update the status file after all clients have completed
        update_status_file_as_completed(comm_round)

        # Save the current_cluster dictionary to a file
        save_cluster_to_file(current_cluster, comm_round)

        debug(f"SERVER>>>>, End of COMMUNICATION ROUND: {comm_round}, Current Cluster: {current_cluster}")

        #main(model, comm_round) #Start FL server
        time.sleep(30)

def fedcohort(FL_round, NUM_CLIENTS, current_cluster, Use_Model_Parameters_for_cohorting):

    """
    Federated Learning process with Federated Cohorting.

    Args:
    - FL_round (int): Total number of communication rounds for the federated learning process.
    - NUM_CLIENTS (int): Total number of clients participating in the federated learning process.
    - current_cluster (dict): Dictionary containing the current clustering of clients. Keys represent cohort identifiers
      and values represent lists of client IDs belonging to each cohort.
    - Use_Model_Parameters_for_cohorting (int): Flag indicating whether to use model parameters for cohorting (1 for True,
      0 for False).

    Returns:
    - None

    Description:
    This function implements the Federated Cohorting algorithm for federated learning. It trains a deep learning
    model collaboratively across multiple clients while considering the clustered nature of the data and handling drift.
    The process involves several communication rounds where clients perform local training on their data and communicate
    model updates to the server. Drift detection and cohort reassignment are performed during certain communication
    rounds to adapt to changes in the data distribution.

    This function performs the following steps:
    1. If 'Use_Model_Parameters_for_cohorting' is set to 1, it updates the 'current_cluster' based on model parameters.
       Otherwise, it uses the provided 'current_cluster'.
    2. Reads server data from a CSV file located at a predefined path.
    3. Splits the data into training and validation sets.
    4. Iterates over communication rounds, performing the following steps in each round:
       a. Initializes a machine learning model.
       b. Creates necessary text files for status checking and updating data drift.
       c. Resets the status file for the current communication round.
       d. Checks the number of cohorts:
          - If there's only one cohort, it performs federated learning on that cohort's clients.
          - If there are multiple cohorts, it performs federated learning on each cohort's clients separately.
       e. If the current communication round is greater than 1, performs drift detection:
          - Identifies drifted and non-drifted clients.
          - Clusters drifted clients to create new cohorts.
       f. Waits for clients to complete their training for the current communication round.
       g. Updates the status file after all clients have completed the round.
       h. Saves the current clustering to a file.
       i. Sleeps for 30 seconds before starting the next communication round.

    Note:
    - The function relies on various helper functions such as 'get_cohorts', 'custom_train_test_split', 'debug',
      'model_initialisation', 'load_previous_model', 'get_ready_clients', 'main', 'save_trained_server_model',
      'server_functions', 'get_cohorts_after_drift', 'wait_for_clients_completion', 'update_status_file_as_not_completed',
      'update_status_file_as_completed', 'save_cluster_to_file', and 'time.sleep'.
    """

    if Use_Model_Parameters_for_cohorting == 1:
        current_cluster = get_cohorts(NUM_CLIENTS)

    debug("Reading server data...")
    server_data = pd.read_csv(BASE_PATH + r"Data\server.csv")

    X_train, X_val, y_train, y_val = custom_train_test_split(server_data)

    for comm_round in range(1, FL_round+1):        
        debug(f"SERVER>>>>, Beginning of COMMUNICATION ROUND: {comm_round}, Current Cluster: {current_cluster}")
        model = model_initialisation()

        # creates all the text files for status checking and updating drift in data.
        create_neccessary_text_files(comm_round) 

        # Reset the communication round status file
        update_status_file_as_not_completed(comm_round)
            
        no_cohorts = len(current_cluster)
        if no_cohorts == 1:
            key, values = list(current_cluster.items())[0]
            debug(f"SERVER>>>> No of cohorts: {no_cohorts}, Performing FL on cohort : {key} with clients {values}")
            get_ready_clients(values)

            if comm_round == 1:
                model = model_initialisation()
            else:
                #load the previous communication round model
                model = load_previous_model(comm_round, key)
                if model is None:
                    model = model_initialisation()
                debug(f"SERVER>>>> Communication round: {comm_round-1} - Model: server_model_round{comm_round-1}.h5")
            
            model = main(X_train, X_val, y_train, y_val, model, comm_round, key, min_fit_clients=NUM_CLIENTS, min_available_clients=NUM_CLIENTS) # Need to pass the parameters as the per the number of clients in the cohort to trigger the FL procress.
            save_trained_server_model(model, comm_round, key)

        elif no_cohorts > 1:
            for key, values in current_cluster.items():
                if comm_round == 1:
                    model = model_initialisation()
                elif comm_round>1:
                    model = load_previous_model(comm_round, key)
                    if model is None:
                        model = model_initialisation()
                debug(f"SERVER>>>> No of cohorts: {no_cohorts}, Performing FL on cohort : {key} with clients {values}")
                get_ready_clients(values)
                #model = load_cohort_model(key, comm_round)
                model = main(X_train, X_val, y_train, y_val, model, comm_round, key, min_fit_clients=len(values), min_available_clients=len(values)) # Need to pass the parameters as the per the number of clients in the cohort to trigger the FL procress.
                save_trained_server_model(model, comm_round, key)

        if comm_round > 1:
            drifted_clients, non_drifted_clients, current_cluster = server_functions(comm_round, current_cluster)
            time.sleep(30)
            if (len(drifted_clients) > 0)  and (len(non_drifted_clients)<NUM_CLIENTS):
                time.sleep(60) # wait till the clients retrain their model

                current_cluster = get_cohorts_after_drift(NUM_CLIENTS, current_cluster, drifted_clients)
            else:
                pass

        #wait for the clients to complete the communication round.  
        wait_for_clients_completion(comm_round, NUM_CLIENTS)

        # Update the status file after all clients have completed
        update_status_file_as_completed(comm_round)

        # Save the current_cluster dictionary to a file
        save_cluster_to_file(current_cluster, comm_round)

        debug(f"SERVER>>>>, End of COMMUNICATION ROUND: {comm_round}, Current Cluster: {current_cluster}")

        #main(model, comm_round) #Start FL server
        time.sleep(30)


def start_server(FL_round, NUM_CLIENTS, alg, current_cluster, Use_Model_Parameters_for_cohorting):

    """
    Start the federated learning server with the specified algorithm.

    Args:
    - FL_round (int): Total number of communication rounds for the federated learning process.
    - NUM_CLIENTS (int): Total number of clients participating in the federated learning process.
    - alg (str): Algorithm to be used for federated learning. Supported values are 'CDA-FedAvg', 'FedDrift', 'Dissonance',
      'Vannila FL', and 'FedCohort'.
    - current_cluster (dict): Dictionary containing the current clustering of clients. Keys represent cohort identifiers
      and values represent lists of client IDs belonging to each cohort.
    - Use_Model_Parameters_for_cohorting (int): Flag indicating whether to use model parameters for cohorting (1 for True,
      0 for False).

    Returns:
    - None

    Description:
    This function starts the federated learning server with the specified algorithm. Depending on the algorithm chosen,
    it invokes the corresponding federated learning function, passing the required parameters.

    Supported Algorithms:
    - CDA-FedAvg: Concept Drift-Aware Federated Averaging.
    - FedDrift: Federated Learning with Drift Detection.
    - Dissonance: Algorithm in testing phase, not available for public use.
    - Vannila FL: Vanilla Federated Learning (Baseline).
    - FedCohort: Federated Learning with Cohorting.

    Note:
    - The function relies on various federated learning functions such as 'cdafedavg', 'feddrift', 'Vannila_FL', and
      'fedcohort', as well as on a UI library 'st' for displaying notifications.
    - The actual implementation of the federated learning algorithms and UI interactions are not provided within this
      documentation.
    """

    if alg == 'CDA-FedAvg':
        cdafedavg(FL_round, NUM_CLIENTS, current_cluster, Use_Model_Parameters_for_cohorting)
    elif alg == 'FedDrift':
        feddrift(FL_round, NUM_CLIENTS, current_cluster, Use_Model_Parameters_for_cohorting)
    elif alg == 'Dissonance':
        #st.toast('The Dissonance algorithm is in the testing phase and not available for public use.', icon='⛔')
        dissonance(FL_round, NUM_CLIENTS, current_cluster, Use_Model_Parameters_for_cohorting)
    elif alg == "Vannila FL":
        Vannila_FL(FL_round, NUM_CLIENTS, current_cluster, Use_Model_Parameters_for_cohorting)
    elif alg == "FedCohort":
        fedcohort(FL_round, NUM_CLIENTS, current_cluster, Use_Model_Parameters_for_cohorting)

# Check if the script is run as the main module
if __name__ == "__main__":

    """
    Entry point of the script when run as the main module.

    This script extracts command-line arguments, processes them, and calls the 'start_server' function with the
    appropriate arguments.

    Command-line Arguments:
    - FL_round (int): Total number of communication rounds for the federated learning process.
    - NUM_CLIENTS (int): Total number of clients participating in the federated learning process.
    - alg (str): Algorithm to be used for federated learning. Supported values are 'CDA-FedAvg', 'FedDrift',
      'Dissonance', 'Vannila FL', and 'FedCohort'.
    - current_cluster_str (str): String representation of the current clustering of clients. It is converted into a
      dictionary.
    - Use_Model_Parameters_for_cohorting (str): String indicating whether to use model parameters for cohorting.
      Accepted values are '1' for True and '0' for False.

    Note:
    - The script expects command-line arguments in a specific order: FL_round, NUM_CLIENTS, alg,
      Use_Model_Parameters_for_cohorting, and current_cluster_str.
    - The script relies on the 'start_server' function to initiate the federated learning server with the specified
      algorithm and parameters.
    """

    # Extract command-line arguments
    FL_round = int(sys.argv[1])
    print(f'Value of FL_ROUND: {FL_round}, TYPE: {type(FL_round)}')
    NUM_CLIENTS = int(sys.argv[2])
    print(f'Value of NUM_CLIENTS: {NUM_CLIENTS}, TYPE: {type(NUM_CLIENTS)}')
    alg = sys.argv[3]
    print(f'Value of ALGORITHM: {alg}, TYPE: {type(alg)}')

    # Process CLUSTER argument to convert it into a dictionary
    current_cluster_str = sys.argv[5:]
    # Extracting the string from the list
    string_representation = current_cluster_str[0]
    # Using ast.literal_eval to safely evaluate the string as a dictionary
    current_cluster = ast.literal_eval(string_representation)
    print(f'Value of CLUSTER: {current_cluster}, TYPE: {type(current_cluster)}')

    # Get the boolean value of model_parameters for cohorting
    Use_Model_Parameters_for_cohorting = sys.argv[4]

    # Call the start_server function with the provided arguments
    start_server(FL_round, NUM_CLIENTS, alg, current_cluster, Use_Model_Parameters_for_cohorting)