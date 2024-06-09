import sys
sys.path.append('../')

from PIL import Image
from Clients.create_clients import create_clients
from Clients.start_clients import start_clients
from Custom_Scripts.constants import BASE_PATH, virtualenv_path, script_path
from Custom_Scripts.activate_venv import trigger_server
import streamlit as st
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

# Emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
st.set_page_config(page_title="Dashboard", page_icon=":hot_springs:", layout="wide")
#st.set_page_config(page_title="Dashboard", page_icon=":hot_springs")

st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
       

        .stDeployButton {display:none;}

        #stDecoration {display:none;}
       
        .block-container {
            padding-top: 1rem;
            padding-bottom: 5rem;
            padding-left: 5rem;
            padding-right: 5rem;
        }
    </style>
""", unsafe_allow_html=True)

# Page Introduction
st.title("Welcome to the Federated Learning Demo! 👋")

col1, col2, col3 = st.columns([1, 1, 1])

# Initialize session state variables
if 'server_status' not in st.session_state:
    st.session_state.server_status = False
if 'server_clicked' not in st.session_state:
    st.session_state.server_clicked = False
if 'clients_status' not in st.session_state:
    st.session_state.clients_status = False
if 'clients_clicked' not in st.session_state:
    st.session_state.clients_clicked = False
if 'cohort_clients' not in st.session_state:
    st.session_state.cohort_clients = {}
if 'cohort_based_on_model_parameters' not in st.session_state:
    st.session_state.cohort_based_on_model_parameters = 0
if 'fl_process_completed' not in st.session_state:
    st.session_state.fl_process_completed = 0
if 'result_status' not in st.session_state:
    st.session_state.result_status = False
if 'drift' not in st.session_state:
    st.session_state.drift = False

def click_server_button():
    st.session_state.server_clicked = True

def click_clients_button():
    st.session_state.clients_clicked = True

def update_fl_process_status():
    st.session_state.fl_process_completed = 1
    remove_drift_file()  # Call the function to remove the drift file

def remove_drift_file():
    drift_file_path = BASE_PATH + "Results\Drift\Json\drift_entry.json"
    if os.path.exists(drift_file_path):
        os.remove(drift_file_path)  # Remove the file
        st.write("Drift data file removed successfully.")
    else:
        st.write("Drift data file not found.")

def save_drift_entry(drift_entry):
    drift_file_path = BASE_PATH + "Results\Drift\Json\\"
    # Check if the directory already exists
    if not os.path.exists(drift_file_path):
        # If not, create the directory
        os.makedirs(drift_file_path)
    # Save the results to a JSON file
    with open(drift_file_path + f'drift_entry.json', 'w') as json_file:
        json.dump(drift_entry, json_file)
        json_file.write('\n')  # Add a newline to separate each iteration


def main():

    NUM_CLIENTS = st.session_state.NUM_CLIENTS if 'NUM_CLIENTS' in st.session_state else 2

    with col1:
        NUM_CLIENTS = st.number_input(label='Enter the number of Clients to include for Federated Learning', min_value=2, max_value=100, step=1, key='NUM_CLIENTS')
        hide_messages = st.toggle("View Messages", value=False)

        if col1.button("Create Clients"):
            status = create_clients(int(NUM_CLIENTS), hide_messages)
            if status:
                st.toast(body='Clients created successfully!', icon='🎉')
            else:
                st.toast(body=f'Unable to create Clients!', icon="🚨")

    with col2:
        alg_choice = st.selectbox(
            "Choose Algorithm to run FL",
            ("CDA-FedAvg", "Dissonance", "FedCohort", "FedDrift", "Vannila FL"),
            index=None,
            placeholder="Select an algorithm...",
        )
        if NUM_CLIENTS > 3:
            cohort_choice = st.selectbox(
                "Choose cohort strategy",
                ("Cohort based on Model Parameters", "Enter cohorts manually"),
                index=None,
                placeholder="Select a strategy...",
            )

            if cohort_choice == "Cohort based on Model Parameters":
                st.session_state.cohort_based_on_model_parameters = 1
            elif cohort_choice == "Enter cohorts manually":
                st.session_state.cohort_based_on_model_parameters = 0
                FL_COHORTS = st.number_input(label='Enter the number of cohorts for FL', min_value=1, max_value=NUM_CLIENTS, step=1)

                if FL_COHORTS > 1:
                    # Dynamic creation of text boxes for entering client ids for each cohort
                    for i in range(FL_COHORTS):
                        cohort_id = st.text_input(f"Enter Client IDs for Cohort {i+1} (comma-separated)", key=f'cohort_{i+1}')
                        if cohort_id:
                            client_ids = [int(client_id.strip()) for client_id in cohort_id.split(",")]
                            st.session_state.cohort_clients[i+1] = client_ids
                    
            else:
                st.session_state.cohort_clients = {1: list(range(1, NUM_CLIENTS + 1))}
        else:
            st.session_state.cohort_clients = {1: list(range(1, NUM_CLIENTS + 1))}


    with col3:
        FL_COMM_ROUNDS = st.number_input(label='Enter the number of communication rounds for FL', min_value=1, max_value=20, step=1)
        
        # Drift Induction Section
        induce_drift = st.checkbox("Induce Drift")

        if induce_drift:
            st.session_state.drift = True
            communication_round = st.text_input(label='Enter the communication rounds for drift induction', value='', help='Example: 1, 2, 3')
            client_ids_input = st.text_input(label='Enter the client IDs for drift induction (comma-separated)', value='', help='Example: 1, 2, 3')
            drift_type = st.selectbox("Select the type of drift", ("Concept Drift", "Covariate Drift"), index= None, placeholder="Choose drift to be induced")        

            # Store results in a dictionary
            drift_entry = {
                'communication_round': communication_round,
                'client_id': client_ids_input,
                'drift_type': drift_type
            }
            
            if st.button("Add Drift"):
                save_drift_entry(drift_entry)
                st.toast(f"{drift_type} will be induced for client {client_ids_input} during communication round {communication_round}", icon="🔥")

        subcol1, subcol2 = col3.columns([1,1])

        if subcol1.button("Start server", on_click=click_server_button):
            if alg_choice:
                st.session_state.server_status = True
                st.toast(f"Initiating FL Server to run using {alg_choice} algorithm.", icon='🏁')
                #st.session_state.server_output_container = ['Hi', 'This is Adarsh']
                trigger_server(FL_round=FL_COMM_ROUNDS, NUM_CLIENTS=NUM_CLIENTS, alg=alg_choice, current_cluster = st.session_state.cohort_clients, Use_Model_Parameters_for_cohorting = st.session_state.cohort_based_on_model_parameters)
            else:
                st.toast(body=f'Please choose an algorithm!', icon="🚨")

        if subcol2.button("Start Clients", on_click=click_clients_button):
            st.session_state.clients_status = True
            st.session_state.server_status = True #TO check if the clients are able to start?
            if alg_choice:
                if st.session_state.server_status:
                    try:
                        st.write("Trying to start clients.")
                        start_clients(alg_choice, FL_COMM_ROUNDS, NUM_CLIENTS, CURRENT_CLUSTER= st.session_state.cohort_clients, DRIFT=st.session_state.drift)
                        st.toast("Federated Learning Completed", icon="🤖")
                        st.session_state.server_status = False
                        st.session_state.result_status = True
                        if st.session_state.clients_status:
                            update_fl_process_status()
                    except:
                        st.toast(body=f'Issue starting Clients.!', icon="🚨")
                else:
                    st.toast(body=f'Please start the server.!', icon="🚨")

    def display_results():
        """
        Display federated learning results in a Streamlit app.

        This function presents a Streamlit app interface with visualizations and data
        related to federated learning results. It includes options to select clients
        and displays images and plots corresponding to the selected clients.

        Parameters:
        None

        Returns:
        None
        """
        st.subheader("Federated Learning Results")

        # Set up columns for layout
        col1, col2, col3 = st.columns([2, 1.5, 2])

        # Display varying clusters over time in the middle column
        with col2:
            visualize_clusters_over_time(FL_COMM_ROUNDS)
            st.caption("Figure displays varying clusters over time (communication rounds)")

        # Generate a list of client options
        options_list = [f'Client{i}' for i in range(1, NUM_CLIENTS + 1)]

        # Allow the user to select multiple clients
        selected_results = st.multiselect("Select Clients", options_list, default=options_list)

        tab1, tab2, tab3 = st.tabs(["Comapre Loss", "Compare Accuracy", "Other graphs"])

        with tab1:

            #col1 -> vannila FL loss graph
            #col2 -> Vannila FL with drift loss graph
            #col3 -> CDA-FedAvg loss graph
            #col4 -> FedDrift loss graph
            #col5 -> Dissonance loss graph

            # Iterate over selected clients and display results
            for selected_result in selected_results:
                st.write(f"Displaying results for {selected_result}")

                # Set up sub-columns for layout
                col1, col2, col3, col4, col5 = st.columns(5)
                try:
                    # Display images and plots for the selected client
                    with col1.container():
                        plot_loss_curve(selected_result, "VannilaFL_results.json")
                        st.caption("Vannila FL model Loss curve (without drift)")

                        # display_image(selected_result, "stacked_bar_plot.jpg")
                        # st.caption("Base model performance")

                    with col2.container():
                        plot_loss_curve(selected_result, "VannilaFL_Drift_results.json")
                        st.caption("Vannila FL model Loss curve (with drift)")

                        # display_image(selected_result, "FL_stacked_bar_plot.jpg")
                        # st.caption("FL Model performance")

                    with col3.container():
                        plot_loss_curve(selected_result, "CDAFedAvg_results.json")
                        st.caption("CDA-FedAvg Loss curve")

                    with col4.container():
                        plot_loss_curve(selected_result, "FedDrift_results.json")
                        st.caption("FedDrift Loss curve")

                    with col5.container():
                        plot_loss_curve(selected_result, "Dissonance_results.json")
                        st.caption("Dissonance Loss curve")
                    
                except Exception as e:
                    # Handle exceptions and display an error message
                    st.write(f"Unable to display results for {selected_result}. Error: {e}")

        with tab2:
            
            # col1 -> base model stacked bar chart graph
            # col2 -> Vannila FL graph
            # col3 -> CDA-FedAvg graph
            # col4 -> FedDrift graph
            # col5 -> Dissonance graph

            # Iterate over selected clients and display results
            for selected_result in selected_results:
                st.write(f"Displaying results for {selected_result}")

                # Set up sub-columns for layout
                col1, col2, col3, col4, col5 = st.columns(5)
                try:
                    # Display images and plots for the selected client
                    with col1.container():
                        display_image(selected_result, "stacked_bar_plot.jpg")
                        st.caption("Base model performance")

                    with col2.container():
                        display_image(selected_result, "FL_stacked_bar_plot.jpg")
                        st.caption("FL Model performance")

                    with col3.container():
                        display_image(selected_result, "CDAFedAvg_stacked_bar_plot.jpg")
                        st.caption("CDA-FedAvg accuracy")

                    with col4.container():
                        display_image(selected_result, "FedDrift_stacked_bar_plot.jpg")
                        st.caption("FedDrift accuracy")

                    with col5.container():
                        display_image(selected_result, "Dissonance_stacked_bar_plot.jpg")
                        st.caption("Dissonance model Performance")
                
                except Exception as e:
                    # Handle exceptions and display an error message
                    st.write(f"Unable to display results for {selected_result}. Error: {e}")            
        with tab3:
            # Define the algorithm names and file names
            algorithm_names = ['CDAFedAvg', 'FedDrift', 'Dissonance', 'FedAvg']
            file_names = ['CDAFedAvg_results.json', 'FedDrift_results.json', 'Dissonance_results.json', 'VannilaFL_Drift_results.json']
            col1, col2 = st.columns(2)
            with col1:
                # Plot the accuracy curves
                plot_accuracy_curve(algorithm_names, file_names)
            with col2:
                #plot combined loss curve
                plot_metric_curve('loss', algorithm_names, file_names)

    def plot_metric_curve(metric_name, algorithm_names, file_names):
        """
        Plot the average metric curve (loss or accuracy) for all clients across different algorithms.

        Parameters:
        - metric_name (str): Name of the metric to plot ('loss' or 'accuracy').
        - algorithm_names (list): List of algorithm names.
        - file_names (list): List of file names for the JSON files.

        Returns:
        None
        """
        plt.figure(figsize=(12, 8))

        max_comm_round = 0
        for algorithm_name, file_name in zip(algorithm_names, file_names):
            avg_metrics = np.zeros(FL_COMM_ROUNDS)  # Assuming max 10 communication rounds
            counts = np.zeros(FL_COMM_ROUNDS)

            for client_num in range(1, NUM_CLIENTS + 1):
                file_path = os.path.join(BASE_PATH, f'Results\client{client_num}\\Json', file_name)
                #file_path = f"C:\Users\inadnl\OneDrive - ABB\Adarsh work\Dissonance\Results\client{client_num}\\Json\{file_name}"
                if os.path.exists(file_path):
                    with open(file_path, 'r') as file:
                        data_lines = file.readlines()

                    for line in data_lines:
                        entry = json.loads(line)
                        comm_round = entry['comm_round']
                        metric_value = entry[metric_name]

                        if comm_round > max_comm_round:
                            max_comm_round = comm_round

                        avg_metrics[comm_round - 1] += metric_value
                        counts[comm_round - 1] += 1

            avg_metrics = np.divide(avg_metrics, NUM_CLIENTS, where=counts != 0)  # Normalize by the number of clients

            # Create a plot
            plt.plot(np.arange(1, len(avg_metrics) + 1), avg_metrics, label=algorithm_name, marker='o', linestyle='-')

        # Set plot title and labels
        plt.title(f'Average {metric_name.capitalize()} across clients when induced drift')
        plt.xlabel('Communication Round')
        plt.ylabel(f'Average {metric_name.capitalize()}')
        plt.grid(True)
        plt.xticks(np.arange(1, max_comm_round + 1))
        plt.legend()
        
        # Display the plot in the Streamlit app
        st.pyplot(plt)



    def plot_accuracy_curve(algorithm_names, file_names):
        """
        Plot the average accuracy curve for all clients across different algorithms.

        Parameters:
        - algorithm_names (list): List of algorithm names.
        - file_names (list): List of file names for the JSON files.

        Returns:
        None
        """
        plt.figure(figsize=(12, 8))

        max_comm_round = 0
        all_avg_accuracies = {}  # Dictionary to store average accuracy for all algorithms

        for algorithm_name, file_name in zip(algorithm_names, file_names):
            avg_accuracies = np.zeros(FL_COMM_ROUNDS)  # Assuming max 10 communication rounds
            counts = np.zeros(FL_COMM_ROUNDS)

            for client_num in range(1, NUM_CLIENTS + 1):
                file_path = os.path.join(BASE_PATH, f'Results\client{client_num}\\Json', file_name)

                if os.path.exists(file_path):
                    with open(file_path, 'r') as file:
                        data_lines = file.readlines()

                    for line in data_lines:
                        entry = json.loads(line)
                        comm_round = entry['comm_round']
                        accuracy = entry['accuracy']

                        if comm_round > max_comm_round:
                            max_comm_round = comm_round

                        avg_accuracies[comm_round - 1] += accuracy
                        counts[comm_round - 1] += 1

            avg_accuracies = np.divide(avg_accuracies, NUM_CLIENTS, where=counts != 0)  # Normalize by the number of clients

            # Store average accuracy for all algorithms
            all_avg_accuracies[algorithm_name] = avg_accuracies

            # Create a plot
            plt.plot(np.arange(1, len(avg_accuracies) + 1), avg_accuracies, label=algorithm_name, marker='o', linestyle='-')

        # Set plot title and labels
        plt.title('Average Accuracy across clients when induced drift')
        plt.xlabel('Communication Round')
        plt.ylabel('Average Accuracy')
        plt.grid(True)
        plt.xticks(np.arange(1, max_comm_round + 1))
        plt.legend()
        
        # Display the plot in the Streamlit app
        st.pyplot(plt)

        # Calculate and display average accuracy for all communication rounds for each algorithm
        for algorithm_name, avg_accuracies in all_avg_accuracies.items():
            average_accuracy = np.mean(avg_accuracies)
            # Display the average accuracy value for each algorithm
            st.write(f"Average accuracy across all communication rounds for {algorithm_name}: {average_accuracy:.4f}")

    def display_image(selected_result, image_name):
        """
        Display an image in a Streamlit app.

        Parameters:
        - selected_result (str): Label representing the selected client.
        - image_name (str): Name of the image file to be displayed.

        Returns:
        None
        """
        # Extract the client number from the label
        client_num = int(selected_result[6:]) if selected_result != 'Server' else 0
        
        # Construct the full file path for the image
        data_path = os.path.join(BASE_PATH, f"Results\client{client_num}\Plots", image_name)

        try:
            # Check if the image file exists
            if os.path.exists(data_path):
                # Open the image file
                data = Image.open(data_path)

                # Display the image in the Streamlit app with column width
                st.image(data, use_column_width=True)
            else:
                # Display a message if the image file is not found
                st.write(f"Unable to display {image_name}. Image not found.")
        except Exception as e:
            # Display an error message if an exception occurs during image display
            st.write(f"Unable to display {image_name}. Error: {e}")

    def plot_loss_from_history_file(client_num, file_path):
        """
        Plot the training and validation loss from a training history JSON file.

        Parameters:
        - client_num (int): The client number.
        - file_path (str): The path to the JSON file containing training history.

        Returns:
        None
        """
        try:
            # Check if the file exists
            if os.path.exists(file_path):
                # Load the training history from the JSON file
                with open(file_path, 'r') as json_file:
                    history = json.load(json_file)

                # Plot the training and validation loss
                plt.figure(figsize=(10, 6))
                plt.plot(history['loss'], label='Train Loss')
                plt.plot(history['val_loss'], label='Validation Loss')
                plt.title(f'Loss for Client {client_num}')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()

                # Display the plot in the Streamlit app
                st.pyplot(plt)
                plt.close()
            else:
                # Display a message if the file is not found
                st.write(f"Unable to plot loss curve. File not found: {file_path}")
        except Exception as e:
            # Display an error message if an exception occurs during plotting
            st.write(f"Unable to plot loss curve. Error: {e}")

    def plot_loss_curve(selected_result, file_name):
        """
        Plot the loss curve for a specific client's federated learning.

        Parameters:
        - selected_result (str): Label representing the selected client.
        - file_name (str): Name of the JSON file containing loss data.

        Returns:
        None
        """
        # Extract the client number from the label
        client_num = int(selected_result[6:]) if selected_result != 'Server' else 0
        
        # Construct the full file path
        base_path = BASE_PATH +  r"Results\\"
        file_path = os.path.join(base_path, f'client{client_num}\\Json', file_name)

        try:
            # Check if the file exists
            if os.path.exists(file_path):
                # Read data from the JSON file
                with open(file_path, 'r') as file:
                    data_lines = file.readlines()

                # Extract communication rounds and corresponding losses
                comm_rounds = []
                losses = []

                for line in data_lines:
                    entry = json.loads(line)
                    comm_rounds.append(entry['comm_round'])
                    losses.append(entry['loss'])

                # Set color based on client number
                color = plt.cm.jet(client_num / NUM_CLIENTS)
                label = f'Client {client_num}'

                # Create a plot
                plt.figure(figsize=(10, 6))
                plt.plot(comm_rounds, losses, label=label, color=color, marker='o', linestyle='-', alpha=0.7)

                # Highlight points where loss in comm_round i is more than i-1 with a red dot
                for round_num in range(1, len(comm_rounds)):
                    if losses[round_num] > losses[round_num - 1]:
                        plt.scatter(comm_rounds[round_num], losses[round_num], color='red', zorder=5, alpha=0.8)

                # Set plot title and labels
                plt.title(f'Loss with FL for {label}')
                plt.xlabel('Communication Round')
                plt.ylabel('Loss')
                plt.legend()

                # Display the plot in the Streamlit app
                st.pyplot(plt)
                plt.close()
            else:
                # Display a message if the file is not found
                st.write(f"Unable to display {file_name}. File not found.")
        except Exception as e:
            # Display an error message if an exception occurs during plotting
            st.write(f"Unable to plot loss curve for {file_name}. Error: {e}")


    def visualize_clusters_over_time(FL_COMM_ROUNDS):
        """
        Visualize the clusters over time.

        Parameters:
        - FL_COMM_ROUNDS (int): Number of communication rounds.

        Returns:
        None
        """
        # Generate a list of JSON file paths for each communication round
        json_files = [os.path.join(BASE_PATH, f'Results\Clusters\current_cluster_round{COMM_ROUND}.json') for COMM_ROUND in range(1, FL_COMM_ROUNDS + 1)]
        
        # Initialize an empty list to store data points for plotting
        data = []

        # Loop through each JSON file and extract data for plotting
        for time_point, json_file in enumerate(json_files, start=1):
            try:
                # Read JSON data from the file
                with open(json_file, 'r') as f:
                    json_data = json.load(f)

                # Extract information from JSON data and append to the data list
                for cluster, clients in json_data.items():
                    data.extend([(time_point, int(cluster), client) for client in clients])
            except Exception as e:
                # Handle any exceptions that may occur during file reading
                st.write(f"Unable to read {json_file}. Error: {e}")

        # Create a DataFrame from the collected data
        df = pd.DataFrame(data, columns=['Time', 'Cluster', 'Client'])

        # Plot cluster visualization over time
        sns.scatterplot(x='Time', y='Client', data=df, hue='Cluster', palette='viridis', s=100, style='Cluster', markers=True, legend='full')
        
        # Add labels and title to the plot
        plt.xlabel('Communication Rounds')
        plt.ylabel('Clients')
        plt.title('Cluster Visualization Over Time')
        
        # Add legend to the plot
        plt.legend(title='Cluster', bbox_to_anchor=(1, 1))
        
        try:
            # Display the plot in the Streamlit app
            st.pyplot(plt)
        except Exception as e:
            # Handle any exceptions that may occur during plot display
            st.write(f"Unable to display cluster visualization. Error: {e}")
        finally:
            # Close the plot to free up resources
            plt.close()




    #Expander to show results (visible only after FL process is completed)
    with st.expander("View Results"):
        # if st.session_state.fl_process_completed == 1:
        #     display_results()
        # else:
        #     st.warning("Please initiate FL training.", icon="⚠️")
        display_results()
    
if __name__ == "__main__":
    main()
