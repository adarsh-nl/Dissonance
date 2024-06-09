from keras.metrics import categorical_accuracy
from keras.models import load_model
import numpy as np
import time
import json
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import sys
sys.path.append('../')

#custom script to get a chunk of data
from Custom_Scripts.constants import BASE_PATH, LOSS_THRESHOLD, SHORT_MEMORY
from Custom_Scripts.chunkofdata import give_me_data_chunk
from Custom_Scripts.debug import debug
from Custom_Scripts.model_architecture import model_initialisation

class Client_fn:
    def __init__(self, client_id, x_train, x_test, y_train,  y_test):

        """
        Initializes a Client_fn object with client-specific attributes.

        Parameters:
        - client_id (int): Unique identifier for the client.
        - x_train (DataFrame): Features of the training set.
        - x_test (DataFrame): Features of the test set.
        - y_train (Series): Labels of the training set.
        - y_test (Series): Labels of the test set.
        
        """
        self.client_id = client_id
        self.prev_x_train = None
        self.prev_y_train = None
        self.prev_x_test = None
        self.prev_y_test = None
        self.curr_x_train = x_train
        self.curr_y_train = y_train
        self.curr_x_test = x_test
        self.curr_y_test = y_test
        self.prev_loss = float('inf')
        self.prev_accuracy = None
        self.curr_loss = None
        self.curr_accuracy = None
        self.my_model = None
        self.other_model = None
        self.cohort_loss_array = None

    def change_data(self, x_train,  x_test, y_train, y_test):

        """
        Updates the client's data to new values.

        Parameters:
        - x_train (DataFrame): New features for training.
        - x_test (DataFrame): New features for testing.
        - y_train (Series): New labels for training.
        - y_test (Series): New labels for testing.
        """

        self.prev_x_train, self.prev_y_train, self.prev_x_test, self.prev_y_test = self.curr_x_train, self.curr_y_train, self.curr_x_test, self.curr_y_test
        self.curr_x_train, self.curr_y_train, self.curr_x_test, self.curr_y_test = x_train, y_train, x_test, y_test

    def evaluate_model_with_newdata(self):

        """
        Evaluates the client's model on the current test set and updates loss and accuracy.

        Returns:
        - Tuple: Current loss and accuracy.
        """

        #predictions = self.my_model.predict(self.curr_x_test)

        # Calculate the loss (typically cross-entropy for classification tasks)
        self.curr_loss, self.curr_accuracy = self.my_model.evaluate(self.curr_x_test, self.curr_y_test)

        #self.curr_accuracy = categorical_accuracy(self.curr_y_test, predictions).numpy().mean()
        return self.curr_loss, self.curr_accuracy

    def check_for_drift(self):

        """
        Checks for data drift by comparing the current loss with the previous loss.

        Returns:
        - str: "Drift in data" if drift is detected, "No Drift" otherwise.
        """

        if (self.curr_loss > (self.prev_loss + LOSS_THRESHOLD)):
            return "Drift in data"
        else:
            return "No Drift"

    def find_score_and_labels(self, matrix):

        """
        Finds the maximum values and indices in a matrix.

        Parameters:
        - matrix (numpy.ndarray): Input matrix.

        Returns:
        - Tuple: Lists of maximum values and indices.
        """

        # max_values = []
        # max_indices = []

        # for row in matrix:
        #     max_value = max(row)
        #     max_index = np.argmax(row)

        #     max_values.append(max_value)
        #     max_indices.append(max_index)

        # return max_values, max_indices
        max_values = np.max(matrix, axis=1)  # Find the maximum value along axis 1
        max_indices = np.argmax(matrix, axis=1)  # Find the index of the maximum value along axis 1

        return max_values, max_indices

    def predict_labels_for_new_data(self, X_val, y_val):

        """
        Predicts labels and confidence scores for new data.

        Parameters:
        - X_val (DataFrame): Features of the new data.
        - y_val (Series): Labels of the new data.

        Returns:
        - Tuple: Lists of confidence scores and predicted labels.
        """

        predictions = self.my_model.predict(X_val)
        confidence_scores, labels = self.find_score_and_labels(predictions)
        return confidence_scores, labels
    
    def update_short_memory(self, queue, new_values):

        """
        Updates a short memory queue with new values, maintaining a fixed size.

        Parameters:
        - queue (collections.deque): Short memory queue.
        - new_values (List): New values to be added to the queue.

        Returns:
        - collections.deque: Updated short memory queue.
        """

        # If adding new values doesn't exceed the SHORT_MEMORY size, simply add them
        if len(queue) + len(new_values) <= SHORT_MEMORY:
            queue.extend(new_values)
        elif len(queue) == 0 and len(new_values) > SHORT_MEMORY:
            queue.extend(new_values[-SHORT_MEMORY:])
        else:
            # Calculate how many older values to remove
            num_to_remove = len(queue) + len(new_values) - SHORT_MEMORY
            for _ in range(num_to_remove):
                queue.popleft()  # Remove from the left (oldest values)
                    # Remove older values to accommodate new values
            queue.extend(new_values)
        
        return queue
        
    def cohort_loss(self, client_id, comm_round, drifted_clients):

        """
        Computes the cohort loss array based on model evaluations for drifted clients.

        Parameters:
        - client_id (int): Unique identifier for the client initiating the cohort loss calculation.
        - comm_round (int): Current communication round.
        - drifted_clients (List): List of clients experiencing data drift.

        Returns:
        - numpy.ndarray: Cohort loss array.
        """
        
        debug(f"Number of drifted clients in communication round: {comm_round}--> {len(drifted_clients)}")
        for i in drifted_clients:
            if (i == self.client_id):
                model = self.my_model
            else:
                # load model for the other clients
                model = load_model(BASE_PATH + f'Results\client{i}\Models\model_round_{comm_round}.h5')

            for j in drifted_clients:
                if (j == self.client_id):
                    #calculate the loss of the current_client model with its respective data    
                    my_loss, my_accuracy = self.my_model.evaluate(self.curr_x_test, self.curr_y_test)

                    #add the loss to its position in the loss array
                    self.cohort_loss_array[i-1][client_id-1] = my_loss

                else:
                    # code for loading data of the client
                    x_test, y_test = give_me_data_chunk(BASE_PATH + f'Data\client{j}.csv')

                    # code for calculating loss
                    my_loss, my_accuracy = model.evaluate(x_test, y_test)

                    # code for saving the array.
                    self.cohort_loss_array[i-1][j-1]=my_loss

                    #print the status of the cliens model and data    
                debug(f"Model: Client {i}, Data: Client {j}, Loss: {my_loss}")

        return self.cohort_loss_array
    
    def distance_function(self, loss_array, drifted_clients, NUM_CLIENTS):

        """
        Computes the distance matrix based on cohort loss values.

        Parameters:
        - loss_array (numpy.ndarray): Cohort loss array.
        - drifted_clients (List): List of clients experiencing data drift.
        - NUM_CLIENTS (int): Total number of clients.

        Returns:
        - numpy.ndarray: Distance matrix.
        """

        D = np.zeros((NUM_CLIENTS, NUM_CLIENTS))

        for i, client_i in enumerate(drifted_clients):
            for j, client_j in enumerate(drifted_clients):
                Lij = loss_array[client_i-1][client_j-1]
                Lii = loss_array[client_i-1][client_i-1]
                Ljj = loss_array[client_j-1][client_j-1]
                Lji = loss_array[client_j-1][client_i-1]

                # Cluster distances D(i, j) ← max(Lij−Lii, Lji−Ljj , 0)
                D[i, j] = max(Lij - Lii, Ljj - Lji, 0)

        return D
    
    def should_i_be_ready(self, client_id):

        """
        Checks if the client should be ready based on a text file.

        Parameters:
        - client_id (int): Unique identifier for the client.

        Returns:
        - bool: True if the client should be ready, False otherwise.
        """

        file_path = BASE_PATH + r'Scripts\Clients\get_ready_clients.txt'
        
        while True:
            try:
                # Read the contents of the text file
                with open(file_path, 'r') as file:
                    content = file.read()

                # Preprocess the content into an array
                client_ids_array = [int(id.strip()) for id in content.split(',')]

                # Check if the client_id is present in the array
                if client_id in client_ids_array:
                    return True
                else:
                    print(f"Client {client_id} is not in the current cohort. Waiting for the next cohort.")
                    time.sleep(5)  # Adjust sleep time as needed

            except FileNotFoundError:
                print(f"File not found at path: {file_path}")
                time.sleep(240)  # Adjust sleep time as needed
            except Exception as e:
                print(f"An error occurred: {e}")
                time.sleep(240)  # Adjust sleep time as needed
                
    def is_communication_round_completed(self, comm_round):

        """
        Checks if a communication round is marked as completed in a status text file.

        Parameters:
        - comm_round (int): Current communication round.

        Returns:
        - bool: True if the communication round is completed, False otherwise.
        """

        file_path = BASE_PATH +'Results\communication_round_status.txt'
        try:
            with open(file_path, 'r') as status_file:
                content = status_file.read()

            return f'Communication round {comm_round} completed' in content

        except FileNotFoundError:
            print(f"File not found at path: {file_path}")
            return False
        except Exception as e:
            print(f"An error occurred: {e}")
            return False
 
def check_if_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        return

def load_previous_client_model(client_id, comm_round):

    """
    Load the model of a specific client from a previous communication round.

    Parameters:
    - client_id (int): Unique identifier for the client.
    - comm_round (int): Communication round number.

    Returns:
    - keras.models.Model: Loaded model for the specified client and communication round.

    Example:
    ```
    client_id = 1
    comm_round = 3
    model = load_previous_client_model(client_id, comm_round)
    ```
    """

    model = load_model(BASE_PATH + f'Results\client{client_id}\Models\\model_round_{comm_round}.h5')
    return model

def signal_status_completion(comm_round):

    """
    Signal the completion status of a communication round by appending a status entry to a text file.

    Parameters:
    - comm_round (int): Communication round number.

    Example:
    ```
    comm_round = 3
    signal_status_completion(comm_round)
    ```
    """

    with open(BASE_PATH + f'Results\completion_status.txt', 'a') as completion_file:
        completion_file.write(f'completed-{comm_round}\n')

def create_results_file(client_id, alg_name):

    """
    Create a directory for the client's results and an empty results.json file.

    Parameters:
    - client_id (int): Unique identifier for the client.

    Example:
    ```
    client_id = 1
    create_results_file(client_id)
    ```

    The function creates a directory specific to the client within the 'Results' folder
    (e.g., 'Results/client1') and an empty 'results.json' file inside that directory.
    """

    directory_path = os.path.join(BASE_PATH, f'Results\\client{client_id}\\Json')
    
    check_if_path_exists(directory_path)
    
    # Create or open the results.json file within the directory
    with open(os.path.join(directory_path, f'{alg_name}_results.json'), 'w') as json_file:
        pass

def save_client_results(client_id, results, alg_name):

    """
    Save the results of a client to a JSON file.

    Parameters:
    - client_id (int): Unique identifier for the client.
    - results (dict): Dictionary containing client results.

    Example:
    ```
    client_id = 1
    results = {'accuracy': 0.85, 'loss': 0.12}
    save_client_results(client_id, results)
    ```
    """

    # Save the results to a JSON file
    with open(BASE_PATH + f'Results\client{client_id}\\Json\\{alg_name}_results.json', 'a') as json_file:
        json.dump(results, json_file)
        json_file.write('\n')  # Add a newline to separate each iteration

def get_a_random_number():

    """
    Generate and return a random number between 0 and 1.

    Returns:
    - float: Random number.

    Example:
    ```
    random_number = get_a_random_number()
    ```
    """

    return np.random.rand()

def save_confidence_scores(client_id, confidence_score_dict):

    """
    Save confidence scores of a client to a JSON file.

    Parameters:
    - client_id (int): Unique identifier for the client.
    - confidence_score_dict (dict): Dictionary containing confidence scores.

    Example:
    ```
    client_id = 1
    confidence_scores = {'class_1': 0.8, 'class_2': 0.6}
    save_confidence_scores(client_id, confidence_scores)
    ```
    """

    with open(BASE_PATH + f'Results\client{client_id}\\Json\\confidence_scores.json', 'a') as json_file:
        json.dump(confidence_score_dict, json_file)
        json_file.write('\n')

def create_confidenceScores_file(client_id):

    """
    Create a directory for the client's confidence scores and an empty confidence_scores.json file.

    Parameters:
    - client_id (int): Unique identifier for the client.

    Example:
    ```
    client_id = 1
    create_confidenceScores_file(client_id)
    ```
    """

    directory_path = os.path.join(BASE_PATH, f'Results\\client{client_id}\\Json')
    
    check_if_path_exists(directory_path)

    with open(directory_path + f'confidence_scores.json', 'w') as json_file:
        pass
    
def get_updated_clusters_from_server(comm_round):

    """
    Load and convert a JSON file containing updated clusters from the server.

    Parameters:
    - comm_round (int): Communication round number.

    Returns:
    - dict: Dictionary with integer keys and corresponding updated clusters.

    Example:
    ```
    comm_round = 5
    updated_clusters = get_updated_clusters_from_server(comm_round)
    ```
    """

    # Load JSON file
    file_path = BASE_PATH + f'Results\Clusters\current_cluster_round{comm_round}.json'
    with open(file_path, 'r') as file:
        data = json.load(file)

    # Convert keys from strings to integers
    converted_cluster = {int(key): value for key, value in data.items()}

    return converted_cluster

def train_client_model_for_cohorting(X_train, X_test, y_train, y_test, client_id):

    """
    Train a client model for cohorted federated learning and save the model along with a performance plot.

    Parameters:
    - X_train (numpy.ndarray): Features of the training set.
    - X_test (numpy.ndarray): Features of the test set.
    - y_train (numpy.ndarray): Labels of the training set.
    - y_test (numpy.ndarray): Labels of the test set.
    - client_id (int): Unique identifier for the client.

    Example:
    ```
    X_train, X_test, y_train, y_test = load_client_data(client_id)
    train_client_model_for_cohorting(X_train, X_test, y_train, y_test, client_id)
    ```
    """

    checkpoint_filepath = BASE_PATH + f'Results\client{client_id}\Models\Checkpoints\\'
    check_if_path_exists(checkpoint_filepath)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    model = model_initialisation()
    history = model.fit(X_train, y_train, 
                        epochs=50, 
                        batch_size=1000, 
                        validation_data=(X_test, y_test),  # Use validation_data for X_test and y_test
                        verbose=1,
                        callbacks=[model_checkpoint_callback])
    
    model.save(BASE_PATH + f"Results\client{client_id}\\Models\\base_model.h5")

    # Save training history as a JSON file
    history_dict = history.history
    history_json = json.dumps(history_dict, indent=4)

    directory_path = BASE_PATH + f"Results\client{client_id}\\Json\\"
    check_if_path_exists(directory_path)
    history_path = os.path.join(directory_path, f"individual_training_history.json")
    with open(history_path, 'w') as json_file:
        json_file.write(history_json)


    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # Save the combined plot
    directory_path = BASE_PATH + f'Results\client{client_id}\\Plots\\'
    check_if_path_exists(directory_path)
    combined_plot_path = os.path.join(directory_path, f'combined_model_plot.jpg')
    plt.savefig(combined_plot_path)

    evaluate_model_and_plot_stacked_bar_chart(model, client_id, X_test, y_test, "stacked_bar_plot.jpg")
    return

def evaluate_model_and_plot_stacked_bar_chart(model, client_id, X_test, Y_test, plot_name):

    """
    Evaluates a ML/DL model and plots a stacked bar chart comparing true labels
    with predicted labels.

    Args:
        model: The trained machine learning model to evaluate.
        client_id: Unique identifier for the client (used for saving the plot).
        X_test: The test data features.
        Y_test: The true labels for the test data.
        plot_name: Name for the generated plot file.

    Returns:
        None. Saves the generated plot and prints evaluation metrics.
    """

    # Model evaluation
    y_true = Y_test
    y_pred = model.predict(X_test)

    # Consider only the first time step for evaluation
    y_true_single_step = y_true[:, 0, :]
    y_pred_single_step = y_pred[:, 0, :]

    # Extract class labels from one-hot encoded format
    true_labels = np.argmax(y_true_single_step, axis=1)
    pred_labels = np.argmax(y_pred_single_step, axis=1)

    # Calculate the number of samples for each class
    unique_classes, total_samples_per_class = np.unique(true_labels, return_counts=True)

    # Calculate the number of correctly predicted samples for each class
    correct_samples_per_class = np.zeros_like(total_samples_per_class)
    for class_label in unique_classes:
        correct_samples_per_class[class_label] = np.sum(np.logical_and(true_labels == class_label,
                                                                       pred_labels == class_label))

    # Calculate accuracy for each class
    accuracies_per_class = correct_samples_per_class / total_samples_per_class.astype(float)
    
    # Print information for debugging
    print(f'True Labels: {true_labels}')
    print(f'Predicted Labels: {pred_labels}')
    print(f'Total Samples per Class: {total_samples_per_class}')
    print(f'Correct Samples per Class: {correct_samples_per_class}')
    print(f'Accuracies per Class: {accuracies_per_class}')

    # Plot the stacked bar chart using seaborn for better color differentiation
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot total samples
    sns.barplot(x=unique_classes, y=total_samples_per_class,
                color='lightgray', edgecolor='black', label='Total Samples', ax=ax)

    # Plot correctly predicted samples
    sns.barplot(x=unique_classes, y=correct_samples_per_class,
                color='green', edgecolor='black', label='Correctly Predicted', ax=ax)

    ax.set_xlabel('Classes')
    ax.set_ylabel('Number of Samples')
    ax.legend()
    plt.title('Stacked Bar Chart - True vs Predicted Classes')

    # Save the combined plot
    stacked_barchart_path = os.path.join(BASE_PATH, f'Results\client{client_id}\\Plots\\{plot_name}')
    plt.savefig(stacked_barchart_path)

    return


