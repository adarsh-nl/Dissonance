#from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow import keras

def preprocess_data(features, labels, n_timesteps):
    n_samples = len(features)
    n_features = features.shape[1]
    
    num_classes = len(np.unique(labels))
    #print(f'Number of classes in the data": {num_classes}')
    
    # Make the total number of samples divisible by n_timesteps
    num_samples_to_keep = n_samples // n_timesteps * n_timesteps
    features = features.iloc[:num_samples_to_keep]
    labels = labels.iloc[:num_samples_to_keep]

    # Reshape features to have shape (n_samples, n_timesteps, n_features)
    features_reshaped = features.values.reshape((-1, n_timesteps, n_features))
    
    labels_reshaped = labels.values.reshape((-1, n_timesteps, 1))
    
    #print(f'Classes in labels reshaped: {np.unique(labels_reshaped)}')
    # One-hot encode the labels
    labels_encoded = keras.utils.to_categorical(labels_reshaped, num_classes=5)
    

    # Store the features and labels as a list of tuples
    data_list = list(zip(features_reshaped, labels_encoded))
    return data_list