"""
author: Adarsh N L

Script to create 2 clients for Federated Learning experiment using the PRONTO dataset
Clients name: Client1, Client2

"""
import numpy as np
import pandas as pd
import os
from pathlib import Path
import streamlit as st

import sys
sys.path.append('../')
from Custom_Scripts.constants import BASE_PATH

def create_data(num_clients):

    """
    Create synthetic client data and save it as CSV files.

    Args:
    - num_clients (int): Number of clients.

    Returns:
    - None

    Example:
    >>> create_data(5)
    """

    data = pd.read_csv(BASE_PATH + 'Data\pronto.csv')
    for i in range (1, num_clients+1):
        samples = np.random.randint(2000, 3500)
        sampled_data = data.groupby('Fault').apply(lambda x: x.sample(n=samples, replace=True)).reset_index(drop=True)
        sampled_data.to_csv(BASE_PATH + f'Data\client{i}.csv', index=False)

def create_clients(num_clients, show_messages = False):

    """
    Dynamically create client scripts and a batch script to start clients.

    Args:
    - num_clients (int): Number of clients to create.
    - show_messages (bool, optional): Whether to display debug messages. Defaults to False.

    Returns:
    - bool: True if successful, False otherwise.

    Example:
    >>> create_clients(5, show_messages=True)

    Description:
    This function dynamically creates client scripts and a batch script to start multiple clients for federated learning.
    It initializes necessary paths, creates client Python scripts and a batch script, and writes the required code for
    client functionality.

    Steps:
    1. Initialize paths and ensure required modules are imported.
    2. Create a directory for client scripts if it doesn't exist.
    3. Create the 'start_clients.py' script to trigger client processes.
    4. Write code to start individual client processes in parallel using Python's multiprocessing module.
    5. Create Python scripts for each client, customizing functionality based on the chosen federated learning algorithm.
    6. Create a batch script ('start_clients.sh') to start all client scripts in parallel.
    7. Return True if successful, False otherwise.

    Note:
    - Ensure the 'BASE_PATH' variable is correctly defined before calling this function.
    - The function relies on external modules and templates for file creation and customization.
    - It provides an option to display debug messages if 'show_messages' is set to True.
    """
    create_data(num_clients)
    try:
        from Custom_Scripts.constants import BASE_PATH
        from Custom_Scripts.debug import debug
    except ImportError:
        print("Error: Unable to import necessary modules.")
        return False
        
    if show_messages:
        debug("Initializing path for clients...")
        progress_bar = st.progress(10, 'Creating Clients')

    path_dir = Path(BASE_PATH + r'Scripts\\Clients')

    # Create the directory if it doesn't exist
    path_dir.mkdir(parents=True, exist_ok=True)

    """
    The code below creates a a dyanamic python file (start_clients.py) that triggers clients via a UI. Still under developement
    """
    name_file = 'start_clients.py'
    file_path = path_dir.joinpath(name_file)

    docstring = """
    Start multiple client processes for federated learning.

    Args:
    - alg (str): Federated learning algorithm to be used by clients.
    - FL_ROUNDS (int): Number of federated learning rounds.
    - NUM_CLIENTS (int): Total number of clients.
    - CURRENT_CLUSTER (dict): Current cluster configuration.
    - DRIFT (bool): Whether drift detection is enabled.

    Returns:
    - None

    Description:
    This function starts multiple client processes for federated learning. It creates separate processes for each client,
    passing the required arguments to the client function. After starting all client processes, it waits for them to complete
    using the 'join' method to ensure synchronization. Finally, it sets an event to allow all processes to start at the same time.

    Steps:
    1. Create an event object for synchronization.
    2. Start a separate process for each client using the 'Process' class from the multiprocessing module.
    3. Pass the federated learning algorithm, round number, number of clients, current cluster configuration, and drift detection flag
       as arguments to each client function.
    4. Use the 'join' method to wait for all client processes to complete.
    5. Set an event to indicate that all processes are ready to start.

    Note:
    - Ensure that client functions (e.g., start_client1, start_client2, etc.) are defined and imported correctly.
    - This function assumes that client functions accept the specified arguments.
    - Clients are started in parallel to leverage multiprocessing capabilities for efficient execution.
    - The function does not return any value but starts client processes to perform federated learning tasks.
    """

    newfile = open(file_path, 'w')
    newfile.write("from multiprocessing import Process, Event\n")
    newfile.close()
    for i in range(1, num_clients +1):
        newfile = open(file_path, 'a')  # Use 'w' mode to create a new file or overwrite an existing one
        newfile.write(f'from Clients.client{i} import start_client{i}\n')
    
    newfile.write("\ndef start_clients(alg, FL_ROUNDS, NUM_CLIENTS, CURRENT_CLUSTER, DRIFT):\n")
    newfile.write('\t"""\n\t' + docstring.strip() + '\n\t"""\n\n')
    newfile.write("\n\te = Event()  # Create event that will be used for synchronization\n")
    for i in range(1, num_clients+1):
        newfile.write(f'\tp{i} = Process(target=start_client{i}, args=(alg, FL_ROUNDS, NUM_CLIENTS, CURRENT_CLUSTER, DRIFT, e))\n\tp{i}.start()\n')
    
    for i in range(1, num_clients+1):
        newfile.write(f"\n\tp{i}.join()")
    
    newfile.write("\n\te.set()  # Set event so all processes can start at the same time")
    
    newfile.close()

    docstring = """
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
    for i in range(1, num_clients + 1):
        name_file = 'client' + str(i) + '.py'
        file_path = path_dir.joinpath(name_file)

        try:
            a = open(BASE_PATH + 'Templates\\template1.txt', "r")
            b = open(BASE_PATH + 'Templates\\template2.txt', "r")

            newfile = open(file_path, 'w')  # Use 'w' mode to create a new file or overwrite an existing one

            for line in a:
                newfile.write(line)

            t1 = 'data = pd.read_csv(BASE_PATH + "Data\\client'
            t2 = str(i)
            t3 = '.csv")'
            newfile.write(t1 + t2 + t3)
            newfile.write("\n")

            for line in b:
                newfile.write(line)

            line = f'\n\ndef start_client{i}(alg, FL_ROUNDS, NUM_CLIENTS, CURRENT_CLUSTER, DRIFT, e):\n\t"""\n\t' + docstring.strip() + f'\n\t"""\n\n\n\tdata = pd.read_csv(BASE_PATH + "Data\client{i}.csv")\n\tX_train, X_test, y_train, y_test = custom_train_test_split(data)\n\tclient_id = create_clientid(os.path.splitext(os.path.basename(__file__))[0])\n\tif alg == "CDA-FedAvg":\n\t\tcdafedavg(client_id, data, X_train, X_test, y_train, y_test, FL_round=FL_ROUNDS)\n\telif alg =="FedDrift":\n\t\tfeddrift(client_id, data, X_train, X_test, y_train, y_test, FL_round=FL_ROUNDS, NUM_CLIENTS=NUM_CLIENTS, current_cluster = CURRENT_CLUSTER)\n\telif alg=="Dissonance":\n\t\tdissonance(client_id, data, X_train, X_test, y_train, y_test, FL_round = FL_ROUNDS, NUM_CLIENTS = NUM_CLIENTS, current_cluster = CURRENT_CLUSTER)\n\telif alg == "Vannila FL":\n\t\tVannila_FL(client_id, data, X_train, X_test, y_train, y_test, FL_round=FL_ROUNDS, NUM_CLIENTS=NUM_CLIENTS, current_cluster=CURRENT_CLUSTER, DRIFT=DRIFT)\n\telif alg == "FedCohort":\n\t\tfedcohort(client_id, data, X_train, X_test, y_train, y_test, FL_round=FL_ROUNDS, NUM_CLIENTS=NUM_CLIENTS, current_cluster = CURRENT_CLUSTER)'
            newfile.write(line)

            a.close()
            newfile.close()
            b.close()

        except Exception as e:
            print(f"Error creating client {i}: {e}")
            return False
        
        if show_messages:
            debug("Client {} created.".format(i))
            # Update progress bar
            progress_percentage = (i*10 )
            progress_bar.progress(progress_percentage)
    # Create a batch script to start the clients
    bat_script_path = BASE_PATH + 'Scripts'
    batch_script_name = 'start_clients.sh'

    batch_script = os.path.join(path_dir, batch_script_name)

    try:
        with open(batch_script, 'w') as batch_file:
            batch_file.write('#!/bin/bash\n')
            for i in range(1, num_clients + 1):
                client_script_name = 'client' + str(i) + '.py'
                batch_file.write(f'echo "starting {client_script_name}"\n')
                batch_file.write(f'python {client_script_name} --partition= {i}' ' &\n')

            batch_file.write(f'\ntrap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM\nwait')

        if show_messages:
            # Final progress update to 100%
            progress_bar.progress(100)
        print(f'Batch script created: {batch_script}')
        return True

    except Exception as e:
        print(f"Error creating batch script: {e}")
        return False

