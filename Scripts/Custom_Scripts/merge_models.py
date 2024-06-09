import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
from keras.models import load_model

#load a module from Custom_Scripts
from Custom_Scripts.model_architecture import model_initialisation
from Custom_Scripts.constants import BASE_PATH

def load_models(client_ids, comm_round):

    """
    Load models from different clients for a specific communication round.

    Parameters:
    - client_ids (list): List of client IDs.
    - comm_round (int): Communication round number.

    Returns:
    - list: List of loaded models.
    """

    models = []
    for client_id in client_ids:
        # Load the entire model (including architecture and weights)
        model = keras.models.load_model(BASE_PATH + f'Results\client{client_id}\Models\model_round_{comm_round}.h5')
        models.append(model)
    return models

def average_models(models):

    """
    Average the weights of a list of models.

    Parameters:
    - models (list): List of models to average.

    Returns:
    - model: A new model with averaged weights.
    """

    # Get the number of models
    num_models = len(models)

    # Get the weights from the first model
    averaged_weights = models[0].get_weights()

    # Sum the weights of all models
    for i in range(1, num_models):
        current_weights = models[i].get_weights()
        averaged_weights = [np.add(averaged_weights[j], current_weights[j]) for j in range(len(averaged_weights))]

    # Average the weights
    averaged_weights = [np.divide(weight, num_models) for weight in averaged_weights]

    # Create a new model with the averaged weights
    averaged_model = model_initialisation()
    averaged_model.set_weights(averaged_weights)

    return averaged_model

def merge_models(client_ids_list, comm_round):

    """
    Merge models from different cohorts for a specific communication round.

    Parameters:
    - client_ids_list (list of lists): List of lists, each containing client IDs in a cohort.
    - comm_round (int): Communication round number.

    Returns:
    - list: List of merged models.
    """

    merged_models = []
    
    for client_ids in client_ids_list:
        # Load individual models
        models = load_models(client_ids, comm_round)
        
        # Average the models
        merged_model = average_models(models)
        
        merged_models.append(merged_model)
    
    return merged_models

def save_new_cohort_model(my_dict, tuples_list, models_list, comm_round):

    """
    Save merged models for each cohort based on a dictionary mapping cohorts to client tuples.

    Parameters:
    - my_dict (dict): Dictionary mapping cohort IDs to client tuples.
    - tuples_list (list): List of tuples representing client IDs.
    - models_list (list): List of models corresponding to the tuples.
    - comm_round (int): Communication round number.
    """

    save_path = BASE_PATH + "Results\server"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for model_tuple, model in zip(tuples_list, models_list):
        for key, values in my_dict.items():
            if set(model_tuple) == set(values):
                model_name = f"server_model_round{comm_round}_cohort{key}.h5"
                model.save(os.path.join(save_path, model_name))
                print(f"Saved model for cohort {key} as {model_name}")

def load_cohort_model(cohort_id, comm_round):

    """
    Load a model for a specific cohort and communication round.

    Parameters:
    - cohort_id (int): Cohort ID.
    - comm_round (int): Communication round number.

    Returns:
    - model or None: Loaded model or None if the model is not found.
    """

    model = None

    if cohort_id == 1:
        model_path = BASE_PATH + f"Results\server\server_model_round{comm_round-1}.h5"
    else:
        # Assuming cohort_id is a key in the dictionary
        model_path = BASE_PATH + f"Results\cohort_models\comm_round_{comm_round-1}_cohort_{cohort_id}_model.h5"

    # Check if the model file exists
    if os.path.exists(model_path):
        model = load_model(model_path)
        print(f"Loaded model for cohort {cohort_id} from {model_path}")
    else:
        print(f"Model for cohort {cohort_id} not found for round {comm_round}")

    return model
