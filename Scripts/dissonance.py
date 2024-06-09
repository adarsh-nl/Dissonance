import tensorflow as tf
import numpy as np
from Custom_Scripts.constants import BASE_PATH
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean

def debug(*args):
    DEBUG = False
    if DEBUG:
        print(*args)

def get_weights(loaded_model):
    # Get the weights of each layer
    model_weights = []
    for layer in loaded_model.layers:
        layer_weights = layer.get_weights()
        model_weights.extend([w.flatten() for w in layer_weights])

    # Concatenate all flattened weights
    model_weights_flat = np.concatenate(model_weights)
    debug(model_weights_flat)
    return model_weights_flat

def get_model_info(model):
    # Loop through the layers and print the shape of each layer's output
    for layer in model.layers:
        layer_name = layer.name
        input_shape = layer.input_shape
        output_shape = layer.output_shape
        print(f"Layer: {layer_name}, Input Shape: {input_shape}, Output Shape: {output_shape}")

# get_model_info(global_model)
        
# def calculate_concept_drift_divergence(local_model_weights, global_model_weights, xi_concept):
#     euclidean_distance = euclidean(local_model_weights, global_model_weights)
#     print(f"Euclidean distance between the global model and the local model are: {euclidean_distance}")
#     concept_drift_divergence = 1 / (1 + np.exp(-xi_concept * euclidean_distance**2))
#     return concept_drift_divergence

def calculate_concept_drift_divergence(local_model_weights, global_model_weights, xi_concept):
    frobenius_norm = np.linalg.norm(local_model_weights - global_model_weights)
    print(f"Frobenius norm distance between the global model and the local model are: {frobenius_norm}")
    concept_drift_divergence = 1 / (1 + np.exp(-xi_concept * frobenius_norm))
    return concept_drift_divergence

def calculate_covariate_drift_divergence(local_model_weights, global_model_weights, xi_covariate):
    cosine_distance = cosine_similarity(local_model_weights.reshape(1, -1), global_model_weights.reshape(1, -1))[0][0]
    #covariate_drift_divergence = (1 + np.tanh(xi_covariate * np.arccos(cosine_distance))) / (np.linalg.norm(local_model_weights) * np.linalg.norm(global_model_weights))
    #return covariate_drift_divergence
    return 1- cosine_distance

def calculate_label_drift_divergence(local_model_weights, global_model_weights, xi_label):
    local_softmax = tf.nn.softmax(local_model_weights)
    global_softmax = tf.nn.softmax(global_model_weights)
    euclidean_distance = euclidean(local_softmax, global_softmax)
    print(f"Euclidean distance between the global model and the local model are: {euclidean_distance}")
    label_drift_divergence = 1 / (1 + np.exp(-xi_label * euclidean_distance**2))
    return label_drift_divergence

def dissonance_metric(local_model_weights, global_model_weights, alpha, beta, gamma, xi_concept, xi_covariate, xi_label):
    # Calculate individual drift metrics
    concept_drift = calculate_concept_drift_divergence(local_model_weights, global_model_weights, xi_concept)
    covariate_drift = calculate_covariate_drift_divergence(local_model_weights, global_model_weights, xi_covariate)
    label_drift = calculate_label_drift_divergence(local_model_weights, global_model_weights, xi_label)

    # Combine individual drift metrics into the Dissonance Metric
    dissonance = alpha * concept_drift + beta * covariate_drift + gamma * label_drift
    return dissonance, concept_drift, covariate_drift, label_drift


global_model = tf.keras.models.load_model(BASE_PATH + "Results\Temp\Server\Models\model.h5")
global_model_weights = get_weights(global_model)

client_model = tf.keras.models.load_model(BASE_PATH + "Results\Temp\client2\Models\model.h5")
client_model_weights = get_weights(client_model)

# Set your hyperparameters
alpha = 1
beta = 1
gamma = 1
xi_concept = 1
xi_covariate = 1
xi_label = 1

# Calculate the dissonance metric
dissonance_value, concept_drift_value, covariate_drift_value, label_drift_value = dissonance_metric(client_model_weights, global_model_weights, alpha, beta, gamma, xi_concept, xi_covariate, xi_label)

print(f'Dissonance Metric: {dissonance_metric}\nConcept Drift Divergence: {concept_drift_value},\nCovariate Drift Divergence: {covariate_drift_value}, Label Drift Divergence: {label_drift_value}')