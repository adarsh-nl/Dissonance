# Federated Learning Configuration Parameters
FL_round = 10  # Number of federated learning rounds
num_rounds = 1  # Number of communication rounds per federated learning round
GLOBAL_MODELS = 1  # Number of global models (typically 1 in centralized federated learning)
NUM_CLIENTS = 10  # Number of participating clients in the federated learning setup
DISTANCE_THRESHOLD = 0.1  # Distance threshold for drift detection
LOSS_THRESHOLD = 0.1  # Loss threshold for drift detection
BASE_PATH = r'C:\Users\inadnl\OneDrive - ABB\Adarsh work\Dissonance\\'  # Base path for project directory

# Paths for Triggering Server
virtualenv_path = r'C:\Adarsh work\venv\ABBProjects'  # Path to the virtual environment
script_path = r'C:\Users\inadnl\OneDrive - ABB\Adarsh work\Dissonance\Scripts\Server\server.py'  # Path to the server script

# Cluster Information
current_cluster = {1: [1,2,3,4,5,6,7,8,9,10]}
# Example clusters:
# current_cluster = {1: [1,2,3,4,5], 2: [6,7,8,9,10]}
# current_cluster = {1: [1,2,3,4,5], 2: [6,7,8], 3: [9,10]}

# Memory Sizes for Clients
LONG_MEMORY = 50000  # Size of the long memory for clients
SHORT_MEMORY = 10000  # Size of the short memory for clients
DELTA = 100  # Delta value for memory management
LAMBDA = 0.05  # Lambda value for memory management
