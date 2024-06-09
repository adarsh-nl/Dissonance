import time  
from Custom_Scripts.constants import BASE_PATH

def convert_to_dict(comm_round):

    """
    Convert client status information from a text file to a dictionary.

    Parameters:
    - comm_round (int): The communication round for which to retrieve client status.

    Returns:
    - dict: A dictionary containing client IDs as keys and their corresponding status as values.
    """

    path = BASE_PATH + fr"Results\client_status\iteration_{comm_round}.txt"
    
    with open(path, "r") as file:
        content = file.readlines()

    drift_data = {}
    
    for line in content:
        # Check if the line is not empty
        if line.strip():
            try:
                # Assuming the content is in the format: (client_id, status)
                client_id, status = map(str.strip, line.strip('()\n').split(','))
                drift_data[int(client_id)] = status
            except Exception as e:
                print(f"Error parsing line: {line}. Error: {e}")

    return drift_data

def find_drifted_clients(drifted_data):

    """
    Identify drifted and non-drifted clients based on their status.

    Parameters:
    - drifted_data (dict): A dictionary containing client IDs as keys and their corresponding drift status as values.

    Returns:
    - tuple: A tuple containing two lists:
        - List of drifted client IDs.
        - List of non-drifted client IDs.
    """

    drifted_clients = []
    non_drifted_clients = []

    for client_id, status in drifted_data.items():
        if status == 'No Drift':
            non_drifted_clients.append(client_id)
        else:
            drifted_clients.append(client_id)


    return drifted_clients, non_drifted_clients

def eval_and_confirmdrfit(client, client_id, comm_round):
    """
    Evaluate the model, check for drift, and confirm the drift status.

    Parameters:
    - client (Client_fn): The client instance containing model and data information.
    - client_id (int): The ID of the client.
    - comm_round (int): The communication round number.

    Returns:
    - tuple: A tuple containing the following information:
        - Loss of the model.
        - Accuracy of the model.
        - List of drifted client IDs.
        - List of non-drifted client IDs.
    """
    if client.my_model is not None and client.my_model.weights:
        if client.curr_x_test is not None and len(client.curr_x_test) > 0:
            # Check for drift and evaluate the model in rounds after the first one
            if comm_round > 1:
                loss, accuracy = client.evaluate_model_with_newdata()
                status = client.check_for_drift()
                client.prev_loss = client.curr_loss

                # Write the client's status to the server's file
                with open(BASE_PATH + f"Results\client_status\iteration_{comm_round}.txt", "a") as file:
                    file.write(f"({client_id}, {status})\n")

                confirmed = False
                while not confirmed:
                    with open(BASE_PATH + f"Results\client_status\iteration_{comm_round}.txt", "r") as file:
                        content = file.read()
                        if f"({client_id}, {status})" in content:
                            confirmed = True
                        else:
                            with open(BASE_PATH + f"Results\client_status\iteration_{comm_round}.txt", "a") as file:
                                file.write(f"({client_id}, {status})\n")

                #wait till all the clients update their status on the file
                time.sleep(5)

                drift_data = convert_to_dict(comm_round)
                drifted_clients, non_drifted_clients = find_drifted_clients(drift_data)
            else:
                return
        else:
            print("curr_x_test is empty or not properly set.")
    else:
        print("my_model is not properly initialized or does not have weights.")

    return loss, accuracy, drifted_clients, non_drifted_clients