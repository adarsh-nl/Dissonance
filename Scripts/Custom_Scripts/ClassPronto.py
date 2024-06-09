import sys
sys.path.append('../')

from Custom_Scripts.debug import debug
from Custom_Scripts.model_architecture import model_initialisation
from Custom_Scripts.preprocess import preprocess_data
from Custom_Scripts.constants import BASE_PATH

import os
import flwr as fl

class Pronto(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test, drift_status = None):

        """
        Pronto: Federated Learning Client Implementation

        Attributes:
        - model (keras.models.Model): The client's local model.
        - x_train (numpy.ndarray): Features of the local training set.
        - y_train (numpy.ndarray): Labels of the local training set.
        - x_test (numpy.ndarray): Features of the local test set.
        - y_test (numpy.ndarray): Labels of the local test set.
        - previous_loss (float): Track the previous loss during training.
        - drift_status (function): Drift detection callback function.
        - directory_path (str): Path to the directory for saving results.

        Methods:
        - get_parameters(): Get the current weights of the local model.
        - fit(parameters, config): Perform local training using received parameters.
        - evaluate(parameters, config): Evaluate the local model using received parameters.
        - save_model(model_path): Save the current model to a specified path.
        - on_fit_end(model_path): Called at the end of the fit operation to save the model.

        Example:
        ```
        model = model_initialisation()
        x_train, y_train, x_test, y_test = preprocess_data()
        client = Pronto(model, x_train, y_train, x_test, y_test)
        ```
        """
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.previous_loss = None  # Track previous loss
        self.drift_status = drift_status  # Drift detection callback function
        self.directory_path = BASE_PATH + 'Results'
    
    def get_parameters(self):

        """
        Get the current weights of the local model.

        Returns:
        - list: List of numpy arrays representing the model weights.

        Example:
        ```
        parameters = client.get_parameters()
        ```
        """

        return self.model.get_weights()

    def fit(self, parameters, config):

        """
        Perform local training using received parameters.

        Parameters:
        - parameters (list): List of numpy arrays representing model weights.
        - config: Configuration object (unused in this implementation).

        Returns:
        - tuple: Tuple containing updated parameters, number of examples used for training,
                 and results dictionary with loss and accuracy.

        Example:
        ```
        updated_params, num_examples_train, results = client.fit(parameters, config)
        ```
        """
        self.model.set_weights(parameters)
        history = self.model.fit(self.x_train, self.y_train, batch_size=32, epochs=10, validation_data=(self.x_test, self.y_test))
        current_loss = history.history["loss"][0]
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": current_loss,  # Use the current loss for results
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):

        """
        Evaluate the local model using received parameters.

        Parameters:
        - parameters (list): List of numpy arrays representing model weights.
        - config: Configuration object (unused in this implementation).

        Returns:
        - tuple: Tuple containing loss, number of examples used for testing,
                 and results dictionary with accuracy.

        Example:
        ```
        loss, num_examples_test, results = client.evaluate(parameters, config)
        ```
        """

        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}
    
    def save_model(self, model_path):

        """
        Save the current model to a specified path.

        Parameters:
        - model_path (str): Path to save the model.

        Example:
        ```
        model_path = BASE_PATH + 'Results/client1/Models/model.h5'
        client.save_model(model_path)
        ```
        """

        # Save the model in the directory path
        self.model.save(model_path)

    def on_fit_end(self, model_path):

        """
        Called at the end of the fit operation to save the model.

        Parameters:
        - model_path (str): Path to save the model.

        Example:
        ```
        model_path = BASE_PATH + 'Results/client1/Models/model.h5'
        client.on_fit_end(model_path)
        ```
        """
        # Call this method at the end of the fit operation
        # It saves the model after each local training iteration
        self.save_model(model_path)