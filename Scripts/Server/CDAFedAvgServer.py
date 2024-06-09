import sys
sys.path.append('../')

from Custom_Scripts.debug import debug
from Custom_Scripts.model_architecture import model_initialisation
from Custom_Scripts.TrainTestSplit import custom_train_test_split
from Custom_Scripts.constants import FL_round, num_rounds, NUM_CLIENTS, current_cluster, BASE_PATH
from Custom_Scripts.server_subModules import wait_for_clients_completion, update_status_file_as_completed, get_ready_clients, create_neccessary_text_files, update_status_file_as_not_completed, load_previous_model
from Custom_Scripts.server_subModules import save_trained_server_model, save_results, fit_config, get_eval_fn, save_cluster_to_file

# import necessary libaries
import time
import flwr as fl
import pandas as pd
from typing import Optional
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

# Filter out Keras warnings about saving models in HDF5 format
warnings.filterwarnings("ignore", category=UserWarning, module=".*tensorflow.keras.*")

# Initialize completion counter
completion_counter = 0

def main(model, comm_round, cohort, min_fit_clients = NUM_CLIENTS, min_available_clients = NUM_CLIENTS):
    debug("Reading server data...")
    server_data = pd.read_csv(BASE_PATH + r"Data\server.csv")

    X_train, X_val, y_train, y_val = custom_train_test_split(server_data)

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        evaluate_fn = get_eval_fn(model, X_val, y_val),
        on_fit_config_fn=fit_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
        fraction_fit = 1.0,
        min_fit_clients  = min_fit_clients,
        min_available_clients = min_available_clients,
    )

    # Start Flower server for five rounds of federated learning
    history_ = fl.server.start_server(server_address="localhost:8083",  strategy=strategy, config=fl.server.ServerConfig(num_rounds=num_rounds))
    
    #save the metrics for plots and analysis
    debug("losses_ centralised", history_.losses_centralized)
    
    save_results(comm_round, cohort, history_)
    
    return model

if __name__ == "__main__":
    for comm_round in range(1, FL_round+1):
        # creates all the text files for status checking and updating drift in data.
        #create_neccessary_text_files(comm_round) 

        # Reset the communication round status file
        update_status_file_as_not_completed(comm_round)

        no_cohorts = len(current_cluster)
        if no_cohorts == 1:
            key, values = list(current_cluster.items())[0]
            debug(f"Server>>>> No of cohorts: {no_cohorts}, Performing FL on cohort : {key} with clients {values}")
            get_ready_clients(values)

            if comm_round == 1:
                model = model_initialisation()
            else:
                #load the previous communication round model
                model = load_previous_model(comm_round, key)
                debug(f"Server>>>> Communication round: {comm_round-1} - Model: server_model_round{comm_round-1}.h5")
            
            model = main(model, comm_round, key) # Need to pass the parameters as the per the number of clients in the cohort to trigger the FL procress.
            save_trained_server_model(model, comm_round, key)

        #wait for the clients to complete the communication round.  
        wait_for_clients_completion(comm_round)

        # Update the status file after all clients have completed
        update_status_file_as_completed(comm_round)
        
        # Save the current_cluster dictionary to a file
        save_cluster_to_file(current_cluster, comm_round)

        #main(model, comm_round) #Start FL server
        time.sleep(30)