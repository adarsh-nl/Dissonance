#Gives access to all the scripts in the directory
import sys
sys.path.append('../')


#boiler plate code for all clients
from Custom_Scripts.debug import debug
from Custom_Scripts.model_architecture import model_initialisation
from Custom_Scripts.ClassPronto import Pronto
from Custom_Scripts.drift import drift
from Custom_Scripts.TrainTestSplit import custom_train_test_split
from Custom_Scripts.ClientID import create_clientid
from Custom_Scripts.ClassClients import Client_fn, load_previous_client_model, signal_status_completion, save_client_results, create_results_file, get_a_random_number, save_confidence_scores, create_confidenceScores_file, get_updated_clusters_from_server, train_client_model_for_cohorting, evaluate_model_and_plot_stacked_bar_chart, check_if_path_exists
from Custom_Scripts.constants import BASE_PATH
from Custom_Scripts.DriftDetection import drift_detection
from Custom_Scripts.DriftAdaption import AdaptDrift
from Custom_Scripts.update_cluster import find_similar_clients, split_clusters
from Custom_Scripts.Eval_and_CheckDrfit import eval_and_confirmdrfit

#import necessary libraries
import os
import time
import tensorflow as tf
import flwr as fl
import numpy as np
import pandas as pd
import warnings
import random
from collections import deque
import matplotlib.pyplot as plt
# Filter out Keras warnings about saving models in HDF5 format
warnings.filterwarnings("ignore", category=UserWarning, module=".*tensorflow.keras.*")

data = pd.read_csv(BASE_PATH + "Data\client4.csv")

X_train, X_test, y_train, y_test = custom_train_test_split(data)
client_id = create_clientid(os.path.splitext(os.path.basename(__file__))[0])
client = Client_fn(client_id, X_train, X_test, y_train, y_test)

#Define constants
DEBUG = True

def main(model, X_train, X_test, y_train, y_test, FL_round):

    """
    Train the federated learning model and start the client for federated learning.

    Args:
    - model (tf.keras.Model): The initial model to be trained.
    - X_train (ndarray): Features of the training data.
    - X_test (ndarray): Features of the test data.
    - y_train (ndarray): Labels of the training data.
    - y_test (ndarray): Labels of the test data.
    - FL_round (int): The current round of federated learning.

    Returns:
    - tf.keras.Model: The trained model.

    Description:
    This function trains the federated learning model using the provided data and configuration. It also starts the client
    for federated learning, saving the trained model for future use.

    Steps:
    1. Set up a model checkpoint callback to save the best model during training.
    2. If it's the first federated learning round:
       - Train the model using the provided data and configuration, saving the best model based on validation accuracy.
    3. Create a Pronto client object and start the client for federated learning.
    4. Define the name of the model file based on the federated learning round.
    5. Create the directory path to save the model within the Results directory.
    6. Save the trained model in the directory path.

    Note:
    - The function relies on external variables such as 'BASE_PATH' and 'client_id' for file paths and client identification.
      Ensure these variables are correctly defined before calling this function.
    - The function utilizes helper functions such as 'check_if_path_exists' and 'Pronto' for directory management and client setup.
    - Ensure the 'BASE_PATH' variable points to the correct directory containing the Results folder.
    """

    global model_name, directory_path

    checkpoint_filepath = BASE_PATH + f'Results\client{client_id}\Models\Checkpoints\\'
    check_if_path_exists(checkpoint_filepath)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    if FL_round == 1:
        history = model.fit(X_train, y_train, 
                            epochs=50, 
                            batch_size=1000, 
                            validation_data=(X_test, y_test),  # Use validation_data for X_test and y_test
                            verbose=1,
                            callbacks=[model_checkpoint_callback])

    # Create client object and start the client
    flclient = Pronto(model, X_train, y_train, X_test, y_test)
    fl.client.start_numpy_client(server_address = "localhost:8083", client=flclient)

    # Get the script's file name (without extension)
    script_name = os.path.splitext(os.path.basename(__file__))[0]

    # Define a dynamic model name based on the FL round
    model_name = f"model_round_{FL_round}.h5"

    # Create the directory path within the Results directory
    directory_path = os.path.join(BASE_PATH + 'Results', script_name)
    directory_path = directory_path +  "\\Models\\"
    
    check_if_path_exists(directory_path)

    flclient.on_fit_end(os.path.join(directory_path, model_name))
    # Save the model in the directory path

    return flclient.model

def cdafedavg(client_id, data, X_train, X_test, y_train, y_test, FL_round, SHORT_MEMORY=1000):

    """
    Perform federated learning using the CDA-FedAvg (Client Drift Aware Federated Averaging) approach on a client.

    Args:
    - client_id (str): Identifier for the client.
    - data (ndarray): Dataset for the client.
    - X_train (ndarray): Features of the training data.
    - X_test (ndarray): Features of the test data.
    - y_train (ndarray): Labels of the training data.
    - y_test (ndarray): Labels of the test data.
    - FL_round (int): Total number of communication rounds for the federated learning process.
    - SHORT_MEMORY (int, optional): Length of short-term memory for storing confidence score values. Default is 1000.

    Returns:
    - None

    Description:
    This function orchestrates federated learning using the CDA-FedAvg approach on a client. It integrates drift detection and
    adaptation mechanisms to handle concept drift during the federated learning process.

    Steps:
    1. Initialize short-term memory queue for storing confidence score values.
    2. Create specific result and confidence scores files for the CDA-FedAvg approach.
    3. Iterate over communication rounds:
       - Load or initialize the model for the communication round.
       - Read the data for training.
       - Wait for server signals and check if the client is ready for training.
       - Train the model using federated learning.
       - Evaluate the model and save results including loss and accuracy for the communication round.
       - Predict confidence scores for new data and store them.
       - Update short-term memory with confidence scores.
       - Detect drift using the short-term memory and adapt model if drift is detected.
       - Signal completion of the communication round.
       - Wait for server synchronization.
    4. Evaluate the final model and generate a stacked bar chart for analysis.

    Note:
    - The function relies on various helper functions such as 'train_client_model_for_cohorting', 'create_results_file',
      'main', 'drift', 'evaluate_model', 'predict_labels_for_new_data', 'save_client_results', 'save_confidence_scores',
      'update_short_memory', 'drift_detection', 'signal_status_completion', and 'evaluate_model_and_plot_stacked_bar_chart'
      for federated learning operations and analysis.
    """

    train_client_model_for_cohorting(X_train, X_test, y_train, y_test, client_id)
    short_memory_queue = deque(maxlen=SHORT_MEMORY) #Initialize short memory Queue for storing the confidence score values of the classifier.
    create_results_file(client_id, "CDAFedAvg")
    create_confidenceScores_file(client_id)
    for comm_round in range(1, FL_round + 1):
        if comm_round > 1:
            model = load_previous_client_model(client_id, comm_round-1)
            #data = drift(client_id, comm_round)
        else:
            model = model_initialisation()

        debug(f"Communication round: {comm_round}, Client {client_id}:  Reading data...")  

        time.sleep(30) # wait till the server says the required clients for training the cohort / batch
        if client.should_i_be_ready(client_id):
            debug(f"Client {client_id} is available for training in the current cohort.")

        client.my_model = main(model, X_train, X_test, y_train, y_test, comm_round)

        if comm_round>1:
            new_data = drift(data, client_id, comm_round, seed = random.choice([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
            X_train, X_test, y_train, y_test = custom_train_test_split(new_data)
            client.change_data(X_train, X_test, y_train, y_test)

            #Evaluate the model to get the metrics
            loss, accuracy = client.evaluate_model_with_newdata()
            
            debug(f"Communication round: {comm_round}, client ID: {client_id}, Loss: {loss}")

            # Store results in a dictionary
            result_entry = {
                'comm_round': comm_round,
                'loss': loss,
                'accuracy': accuracy
            }
            save_client_results(client_id, result_entry, "CDAFedAvg")

            confidence_scores, labels = client.predict_labels_for_new_data(X_test, y_test)

            # Convert NumPy array of floats to a list of regular Python floats
            confidence_scores_serializable = [float(score) for score in confidence_scores.flatten()]

            confidence_scores_dict = {
                'comm_round': comm_round,
                'Confidence Scores': confidence_scores_serializable
            }
            save_confidence_scores(client_id, confidence_scores_dict)


            short_memory_queue = client.update_short_memory(short_memory_queue, confidence_scores)

            #random_number = get_a_random_number()
            if np.exp(-2 * short_memory_queue[len(short_memory_queue)-1]).any() > get_a_random_number():
                if drift_detection(client_id, comm_round, list(short_memory_queue)):
                    debug(f'Communication round: {comm_round}, client ID: {client_id}, Drift detected in data')
                    #Get data for training
                    data = drift(data, client_id, comm_round,size = 500)
                    client.my_model = AdaptDrift(X_train, X_test, y_train, y_test) #train model with new data
                else:
                    pass

        # Signal completion in the completion file
        signal_status_completion(comm_round)
        # Check if the communication round is completed
        while not client.is_communication_round_completed(comm_round):
            print(f"Client {client_id} is waiting for the communication round {comm_round} to be completed.")
            time.sleep(5)  # Adjust sleep time as needed

        debug(f"Client ID: {client_id}, Completed Communication round: {comm_round}")
        
        # Wait till the server starts
        time.sleep(10)

    evaluate_model_and_plot_stacked_bar_chart(client.my_model, client_id, X_test, y_test, "CDAFedAvg_stacked_bar_plot.jpg")

def feddrift(client_id, data, X_train, X_test, y_train, y_test, FL_round, NUM_CLIENTS, current_cluster):
    """
    Perform federated learning with drift detection and adaptation on a client using the FedDrift approach.

    Args:
    - client_id (str): Identifier for the client.
    - data (ndarray): Dataset for the client.
    - X_train (ndarray): Features of the training data.
    - X_test (ndarray): Features of the test data.
    - y_train (ndarray): Labels of the training data.
    - y_test (ndarray): Labels of the test data.
    - FL_round (int): Total number of communication rounds for the federated learning process.
    - NUM_CLIENTS (int): Total number of clients participating in the federated learning process.
    - current_cluster (dict): Dictionary containing the current clustering of clients.

    Returns:
    - None

    Description:
    This function orchestrates federated learning with drift detection and adaptation using the FedDrift approach on a client.
    It trains a model, evaluates performance, handles drift detection, adapts to drift if detected, and communicates with the
    server to synchronize communication rounds.

    Steps:
    1. Initialize a results file specific to the FedDrift approach for the client.
    2. Iterate over communication rounds:
       - Load or initialize the model for the communication round.
       - Read the data for training.
       - Wait for server signals and check if the client is ready for training.
       - Train the model using federated learning.
       - Check for drift and handle it if detected:
         - Reorganize clusters based on drifted clients.
         - Compute loss array for drifted clients and save it.
         - Compute distance metric for merging similar models.
         - Find similar clients based on the distance metric.
         - Sleep until the server merges models if required.
       - Save results including loss and accuracy for the communication round.
       - Signal completion of the communication round.
       - Wait for server synchronization and update the current cluster.
    3. Evaluate the final model and generate a stacked bar chart for analysis.

    Note:
    - The function relies on various helper functions such as 'train_client_model_for_cohorting', 'create_results_file',
      'main', 'drift', 'eval_and_confirmdrfit', 'signal_status_completion', 'get_updated_clusters_from_server', and
      'evaluate_model_and_plot_stacked_bar_chart' for federated learning operations and analysis.
    """
    train_client_model_for_cohorting(X_train, X_test, y_train, y_test, client_id)
    create_results_file(client_id, "FedDrift")
    for comm_round in range(1, FL_round + 1):
        debug(f"Client ID: {client_id}, Beginning of COMMUNICATION ROUND: {comm_round}, Current Cluster: {current_cluster}")
        if comm_round > 1:
            model = load_previous_client_model(client_id, comm_round-1)
            #data = drift(client_id, comm_round)
        else:
            model = model_initialisation()

        debug(f"Communication round: {comm_round}, Client {client_id}:  Reading data...")  
        #X_train, X_test, y_train, y_test = custom_train_test_split(data)

        time.sleep(30) # wait till the server says the required clients for training the cohort / batch
        if client.should_i_be_ready(client_id):
            debug(f"Client {client_id} is available for training in the current cohort.")

        model = main(model, X_train, X_test, y_train, y_test, comm_round)

        time.sleep(10)
        client.my_model = model
        

        if comm_round>1:
            data = drift(data, client_id, comm_round, seed = random.choice([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
            #data = drift(client_id, comm_round)
            X_train, X_test, y_train, y_test = custom_train_test_split(data)
            client.change_data(X_train, X_test, y_train, y_test)

            loss, accuracy, drifted_clients, non_drifted_clients = eval_and_confirmdrfit(client, client_id, comm_round)
            
            debug(f"Communication round: {comm_round}, client ID: {client_id}, Loss: {loss}")

            # Store results in a dictionary
            result_entry = {
                'comm_round': comm_round,
                'loss': loss,
                'accuracy': accuracy
            }
            save_client_results(client_id, result_entry, "FedDrift")

            current_cluster = split_clusters(current_cluster, drifted_clients)
            debug(f"Client {client_id}:\nDrifted Clients: {drifted_clients}\n Non-Drifted Clients: {non_drifted_clients}")

            if client_id in drifted_clients:
                client.cohort_loss_array = [[0]*NUM_CLIENTS]*(NUM_CLIENTS) #Each drifted client is in own cohort now. +1 corresponds to already existing cohort
                cohort_loss = client.cohort_loss(client_id, comm_round, drifted_clients)

                #save the loss array
                np.save(BASE_PATH + f'Results\client{client_id}\comm_round{comm_round}_loss',cohort_loss)

                #distance metric for merging similar models.
                distance_array = client.distance_function(cohort_loss, drifted_clients, NUM_CLIENTS)

                #save the distance array for the server to find the similar models for merging.
                np.save(BASE_PATH + f'Results\DistanceMetric\communication_round{comm_round}_distancemetric', distance_array)

                #call the find_similar_clients function.
                current_cluster, similar_clients = find_similar_clients(drifted_clients, distance_array, current_cluster)

                time.sleep(60) #Sleep until the server merges the models if required
            elif (len(non_drifted_clients) == NUM_CLIENTS) or (len(drifted_clients)==0):
                pass
            else:
                debug(f"Client {client_id}>>>>, Hibernating!, until the chaos is swiped off.")
                time.sleep(60)
                debug(f"Clinet {client_id}>>>>, back from the hibernation")
        
        # Signal completion in the completion file
        signal_status_completion(comm_round)
        
        # Check if the communication round is completed
        while not client.is_communication_round_completed(comm_round):
            print(f"Client {client_id} is waiting for the communication round {comm_round} to be completed.")
            time.sleep(1)  # Adjust sleep time as needed
        
        # Wait till the server starts
        time.sleep(30)
        current_cluster = get_updated_clusters_from_server(comm_round)
        debug(f"Client ID: {client_id}, End of COMMUNICATION ROUND: {comm_round}, Updated cluster: {current_cluster}")
        
    evaluate_model_and_plot_stacked_bar_chart(client.my_model, client_id, X_test, y_test, "FedDrift_stacked_bar_plot.jpg")    
        
def dissonance(client_id, data, X_train, X_test, y_train, y_test, FL_round, NUM_CLIENTS, current_cluster):
    """
    Perform federated learning with the Dissonance algorithm for a single client.

    Args:
    - client_id (str): Unique identifier for the client.
    - data (pd.DataFrame): Dataset for federated learning.
    - X_train (pd.DataFrame): Features of the training data.
    - X_test (pd.DataFrame): Features of the testing data.
    - y_train (pd.Series): Labels of the training data.
    - y_test (pd.Series): Labels of the testing data.
    - FL_round (int): Number of federated learning rounds.
    - NUM_CLIENTS (int): Total number of clients participating in federated learning.
    - current_cluster (dict): Current cluster configuration.

    Returns:
    - None

    Description:
    This function performs federated learning with the Dissonance algorithm for a single client. The Dissonance algorithm
    focuses on detecting drift in the data and adapting the model accordingly. The process involves training a model for each
    communication round, evaluating it, detecting drift, retraining the model if necessary, and signaling completion to the server.

    Steps:
    1. Initialize necessary components such as model and result files.
    2. Loop through each communication round:
        - Load the previous model or initialize a new one.
        - Read the data.
        - Wait for the server to indicate readiness.
        - Train the model on the client's data.
        - Handle drift detection and retraining if necessary.
        - Signal completion to the server.
    3. Evaluate the final model and generate visualizations.

    Note:
    - The function assumes the availability of various helper functions such as drift detection, model adaptation, evaluation,
      and signaling.
    - Drift detection and retraining are crucial steps in the Dissonance algorithm to ensure model adaptation to changing data
      distributions.
    - The Dissonance algorithm aims to maintain model performance despite data drifts and variations.
    """
    DRIFT = False
    train_client_model_for_cohorting(X_train, X_test, y_train, y_test, client_id)
    create_results_file(client_id, "Dissonance")
    for comm_round in range(1, FL_round + 1):
        debug(f"Client ID: {client_id}, Beginning of COMMUNICATION ROUND: {comm_round}, Current Cluster: {current_cluster}")
        if comm_round > 1 and not DRIFT:
            model = load_previous_client_model(client_id, comm_round-1)
            debug(f"Communication round: {comm_round}, Client {client_id}:  Model from communication round {comm_round-1} loaded") 
        elif DRIFT:
            model = model_initialisation()
            debug(f"Communication round: {comm_round}, Client {client_id}:  New model initialized") 
        else:
            model = model_initialisation()
            debug(f"Communication round: {comm_round}, Client {client_id}:  New model initialized") 
            
        debug(f"Communication round: {comm_round}, Client {client_id}:  Reading data...")  
        #X_train, X_test, y_train, y_test = custom_train_test_split(data)

        time.sleep(30) # wait till the server says the required clients for training the cohort / batch
        if client.should_i_be_ready(client_id):
            debug(f"Client {client_id} is available for training in the current cohort.")

        model = main(model, X_train, X_test, y_train, y_test, comm_round)

        time.sleep(10)
        client.my_model = model

        if comm_round>1:
            data = drift(data, client_id, comm_round, seed = random.choice([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
            #data = drift(client_id, comm_round)
            X_train, X_test, y_train, y_test = custom_train_test_split(data)
            client.change_data(X_train, X_test, y_train, y_test)

            loss, accuracy, drifted_clients, non_drifted_clients = eval_and_confirmdrfit(client, client_id, comm_round)
            
            debug(f"Communication round: {comm_round}, client ID: {client_id}, Loss: {loss}")

            # Store results in a dictionary
            result_entry = {
                'comm_round': comm_round,
                'loss': loss,
                'accuracy': accuracy
            }
            save_client_results(client_id, result_entry, "Dissonance")

            if client_id in drifted_clients:
                DRIFT = True
                # the client must retrain their model and save it for the server to use it for cohorting. 

                model = AdaptDrift(X_train, X_test, y_train, y_test, model)
                model.save(BASE_PATH + f"Results\client{client_id}\Models\\client{client_id}_retrained_model.h5")

                time.sleep(60) #Wait till the server looks at the model and cohorts.
            elif (len(non_drifted_clients) == NUM_CLIENTS) or (len(drifted_clients)==0):
                pass
            else:
                debug(f"Client {client_id}>>>>, Hibernating!, until the chaos is swiped off.")
                time.sleep(60)
                debug(f"Clinet {client_id}>>>>, back from the hibernation")
        
        # Signal completion in the completion file
        signal_status_completion(comm_round)
        
        # Check if the communication round is completed
        while not client.is_communication_round_completed(comm_round):
            print(f"Client {client_id} is waiting for the communication round {comm_round} to be completed.")
            time.sleep(1)  # Adjust sleep time as needed
        
        # Wait till the server starts
        time.sleep(30)
        current_cluster = get_updated_clusters_from_server(comm_round)
        debug(f"Client ID: {client_id}, End of COMMUNICATION ROUND: {comm_round}, Updated cluster: {current_cluster}")
        
    evaluate_model_and_plot_stacked_bar_chart(client.my_model, client_id, X_test, y_test, "Dissonance_stacked_bar_plot.jpg")      

def Vannila_FL(client_id, data, X_train, X_test, y_train, y_test, FL_round, NUM_CLIENTS, current_cluster, DRIFT):

    """
    Perform federated learning using the Vanilla Federated Learning (Vannila FL) approach on a client.

    Args:
    - client_id (str): Identifier for the client.
    - data (ndarray): Dataset for the client.
    - X_train (ndarray): Features of the training data.
    - X_test (ndarray): Features of the test data.
    - y_train (ndarray): Labels of the training data.
    - y_test (ndarray): Labels of the test data.
    - FL_round (int): Total number of communication rounds for the federated learning process.
    - NUM_CLIENTS (int): Total number of clients participating in the federated learning process.
    - current_cluster (dict): Dictionary containing the current clustering of clients.
    - DRIFT (bool): Indicates whether drift detection and handling is enabled.

    Returns:
    - None

    Description:
    This function orchestrates federated learning using the Vanilla Federated Learning (Vannila FL) approach on a client.
    It trains a model, evaluates performance, handles drift detection (if enabled), and communicates with the server to
    synchronize communication rounds.

    Steps:
    1. Initialize a results file for the client based on whether drift detection is enabled.
    2. Iterate over communication rounds:
       - Load or initialize the model for the communication round.
       - Read the data for training.
       - Wait for server signals and check if the client is ready for training.
       - Train the model using federated learning.
       - Check for drift and retrain the model if necessary (if drift detection is enabled).
       - Save results including loss and accuracy for the communication round.
       - Signal completion of the communication round.
       - Wait for server synchronization and update the current cluster.
    3. Evaluate the final model and generate a stacked bar chart for analysis.

    Note:
    - The function relies on various helper functions such as 'train_client_model_for_cohorting', 'create_results_file',
      'main', 'drift', 'eval_and_confirmdrfit', 'signal_status_completion', 'get_updated_clusters_from_server', and
      'evaluate_model_and_plot_stacked_bar_chart' for federated learning operations and analysis.
    """

    train_client_model_for_cohorting(X_train, X_test, y_train, y_test, client_id)
    if DRIFT:
        create_results_file(client_id, "VannilaFL_Drift")
    else:
        create_results_file(client_id, "VannilaFL")
    for comm_round in range(1, FL_round + 1):
        debug(f"Client ID: {client_id}, Beginning of COMMUNICATION ROUND: {comm_round}, Current Cluster: {current_cluster}")
        if comm_round > 1:
            model = load_previous_client_model(client_id, comm_round-1)
            #data = drift(client_id, comm_round)
        else:
            model = model_initialisation()

        debug(f"Communication round: {comm_round}, Client {client_id}:  Reading data...")  
        #X_train, X_test, y_train, y_test = custom_train_test_split(data)

        time.sleep(30) # wait till the server says the required clients for training the cohort / batch
        if client.should_i_be_ready(client_id):
            debug(f"Client {client_id} is available for training in the current cohort.")

        model = main(model, X_train, X_test, y_train, y_test, comm_round)

        time.sleep(10)
        client.my_model = model

        if comm_round>1:
            data = drift(data, client_id, comm_round, seed = random.choice([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
            #data = drift(client_id, comm_round)
            X_train, X_test, y_train, y_test = custom_train_test_split(data)
            client.change_data(X_train, X_test, y_train, y_test)

            loss, accuracy, drifted_clients, non_drifted_clients = eval_and_confirmdrfit(client, client_id, comm_round)
            
            debug(f"Communication round: {comm_round}, client ID: {client_id}, Loss: {loss}")

            # Store results in a dictionary
            result_entry = {
                'comm_round': comm_round,
                'loss': loss,
                'accuracy': accuracy
            }
            if DRIFT:
                save_client_results(client_id, result_entry, "VannilaFL_Drift")
            else:
                save_client_results(client_id, result_entry, "VannilaFL")

        # Signal completion in the completion file
        signal_status_completion(comm_round)
        
        # Check if the communication round is completed
        while not client.is_communication_round_completed(comm_round):
            print(f"Client {client_id} is waiting for the communication round {comm_round} to be completed.")
            time.sleep(1)  # Adjust sleep time as needed
        
        # Wait till the server starts
        time.sleep(30)
        current_cluster = get_updated_clusters_from_server(comm_round)
        debug(f"Client ID: {client_id}, End of COMMUNICATION ROUND: {comm_round}, Updated cluster: {current_cluster}")

    evaluate_model_and_plot_stacked_bar_chart(client.my_model, client_id, X_test, y_test, "FL_stacked_bar_plot.jpg")

def fedcohort(client_id, data, X_train, X_test, y_train, y_test, FL_round, NUM_CLIENTS, current_cluster):

    """
    Perform federated learning with cohorting strategy on a client.

    Args:
    - client_id (str): Identifier for the client.
    - data (ndarray): Dataset for the client.
    - X_train (ndarray): Features of the training data.
    - X_test (ndarray): Features of the test data.
    - y_train (ndarray): Labels of the training data.
    - y_test (ndarray): Labels of the test data.
    - FL_round (int): Total number of communication rounds for the federated learning process.
    - NUM_CLIENTS (int): Total number of clients participating in the federated learning process.
    - current_cluster (dict): Dictionary containing the current clustering of clients.

    Returns:
    - None

    Description:
    This function orchestrates federated learning with a cohorting strategy on a client. It trains a model, evaluates
    performance, handles drift detection, and communicates with the server to synchronize communication rounds.

    Steps:
    1. Initialize a results file for the client.
    2. Iterate over communication rounds:
       - Load or initialize the model for the communication round.
       - Read the data for training.
       - Wait for server signals and check if the client is ready for training.
       - Train the model using federated learning.
       - Check for drift and retrain the model if necessary.
       - Save results including loss and accuracy for the communication round.
       - Signal completion of the communication round.
       - Wait for server synchronization and update the current cluster.
    3. Evaluate the final model and generate a stacked bar chart for analysis.

    Note:
    - The function relies on various helper functions such as 'train_client_model_for_cohorting',
      'create_results_file', 'main', 'drift', 'eval_and_confirmdrfit', 'AdaptDrift', 'signal_status_completion',
      'get_updated_clusters_from_server', and 'evaluate_model_and_plot_stacked_bar_chart' for federated learning
      operations and analysis.
    """

    train_client_model_for_cohorting(X_train, X_test, y_train, y_test, client_id)
    create_results_file(client_id, "FedCohort")
    for comm_round in range(1, FL_round + 1):
        debug(f"Client ID: {client_id}, Beginning of COMMUNICATION ROUND: {comm_round}, Current Cluster: {current_cluster}")
        if comm_round > 1:
            model = load_previous_client_model(client_id, comm_round-1)
            debug(f"Communication round: {comm_round}, Client {client_id}:  Model from communication round {comm_round-1} loaded") 
        else:
            model = model_initialisation()
            debug(f"Communication round: {comm_round}, Client {client_id}:  New model initialized") 
            
        debug(f"Communication round: {comm_round}, Client {client_id}:  Reading data...")  
        #X_train, X_test, y_train, y_test = custom_train_test_split(data)

        time.sleep(30) # wait till the server says the required clients for training the cohort / batch
        if client.should_i_be_ready(client_id):
            debug(f"Client {client_id} is available for training in the current cohort.")

        model = main(model, X_train, X_test, y_train, y_test, comm_round)

        time.sleep(10)
        client.my_model = model
        

        if comm_round>1:
            data = drift(data, client_id, comm_round, seed = random.choice([10, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
            #data = drift(client_id, comm_round)
            X_train, X_test, y_train, y_test = custom_train_test_split(data)
            client.change_data(X_train, X_test, y_train, y_test)

            loss, accuracy, drifted_clients, non_drifted_clients = eval_and_confirmdrfit(client, client_id, comm_round)
            
            debug(f"Communication round: {comm_round}, client ID: {client_id}, Loss: {loss}")

            # Store results in a dictionary
            result_entry = {
                'comm_round': comm_round,
                'loss': loss,
                'accuracy': accuracy
            }
            save_client_results(client_id, result_entry, "FedCohort")

            # current_cluster = split_clusters(current_cluster, drifted_clients)
            # debug(f"Client {client_id}:\nDrifted Clients: {drifted_clients}\n Non-Drifted Clients: {non_drifted_clients}")

            if client_id in drifted_clients:
                # the client must retrain their model and save it for the server to use it for cohorting. 

                model = AdaptDrift(X_train, X_test, y_train, y_test, model)
                model.save(BASE_PATH + f"Results\client{client_id}\Models\\client{client_id}_retrained_model.h5")

                time.sleep(60) #Wait till the server looks at the model and cohorts.
            elif (len(non_drifted_clients) == NUM_CLIENTS) or (len(drifted_clients)==0):
                pass
            else:
                debug(f"Client {client_id}>>>>, Hibernating!, until the chaos is swiped off.")
                time.sleep(60)
                debug(f"Clinet {client_id}>>>>, back from the hibernation")
        
        # Signal completion in the completion file
        signal_status_completion(comm_round)
        
        # Check if the communication round is completed
        while not client.is_communication_round_completed(comm_round):
            print(f"Client {client_id} is waiting for the communication round {comm_round} to be completed.")
            time.sleep(1)  # Adjust sleep time as needed
        
        # Wait till the server starts
        time.sleep(30)
        current_cluster = get_updated_clusters_from_server(comm_round)
        debug(f"Client ID: {client_id}, End of COMMUNICATION ROUND: {comm_round}, Updated cluster: {current_cluster}")
        
    evaluate_model_and_plot_stacked_bar_chart(client.my_model, client_id, X_test, y_test, "FedCohort_stacked_bar_plot.jpg")      

def start_client4(alg, FL_ROUNDS, NUM_CLIENTS, CURRENT_CLUSTER, DRIFT, e):
	"""
	Start the client process for federated learning.

    Args:
    - alg (str): Federated learning algorithm to be used by the client.
    - FL_ROUNDS (int): Number of federated learning rounds.
    - NUM_CLIENTS (int): Total number of clients.
    - CURRENT_CLUSTER (dict): Current cluster configuration.
    - DRIFT (bool): Whether drift detection is enabled.
    - e (multiprocessing.Event): Event object for synchronization.

    Returns:
    - None

    Description:
    This function is responsible for initializing and starting the federated learning process for a single client.
    It reads the data for the client from a CSV file, splits it into training and testing sets, and then calls the appropriate
    federated learning algorithm based on the provided algorithm type (alg).

    Supported Algorithms:
    - CDA-FedAvg: Constrained Data Accumulation FedAvg
    - FedDrift: Federated Learning with Drift Detection
    - Dissonance: Placeholder for a custom federated learning algorithm
    - Vannila FL: Vanilla Federated Learning
    - FedCohort: Federated Learning with Cohorting

    Note:
    - Ensure that the CSV file containing the client data is stored at the specified path.
    - The function assumes that the federated learning algorithms (e.g., cdafedavg, feddrift, etc.) are correctly defined and imported.
    - The function is designed to be used as a separate process within a multiprocessing environment.
	"""


	data = pd.read_csv(BASE_PATH + "Data\client4.csv")
	X_train, X_test, y_train, y_test = custom_train_test_split(data)
	client_id = create_clientid(os.path.splitext(os.path.basename(__file__))[0])
	if alg == "CDA-FedAvg":
		cdafedavg(client_id, data, X_train, X_test, y_train, y_test, FL_round=FL_ROUNDS)
	elif alg =="FedDrift":
		feddrift(client_id, data, X_train, X_test, y_train, y_test, FL_round=FL_ROUNDS, NUM_CLIENTS=NUM_CLIENTS, current_cluster = CURRENT_CLUSTER)
	elif alg=="Dissonance":
		dissonance(client_id, data, X_train, X_test, y_train, y_test, FL_round = FL_ROUNDS, NUM_CLIENTS = NUM_CLIENTS, current_cluster = CURRENT_CLUSTER)
	elif alg == "Vannila FL":
		Vannila_FL(client_id, data, X_train, X_test, y_train, y_test, FL_round=FL_ROUNDS, NUM_CLIENTS=NUM_CLIENTS, current_cluster=CURRENT_CLUSTER, DRIFT=DRIFT)
	elif alg == "FedCohort":
		fedcohort(client_id, data, X_train, X_test, y_train, y_test, FL_round=FL_ROUNDS, NUM_CLIENTS=NUM_CLIENTS, current_cluster = CURRENT_CLUSTER)