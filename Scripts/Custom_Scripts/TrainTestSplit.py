import sys
sys.path.append('../')
from Custom_Scripts.preprocess import preprocess_data

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
    
def custom_train_test_split(data):
    features = data.iloc[:, :-1]
    labels = data.iloc[:, -1]
    
    # One-hot encode the labels
    num_classes = len(np.unique(labels))
    labels_encoded = keras.utils.to_categorical(labels, num_classes=num_classes)

    # Z-scale normalize the features column-wise
    scaler = StandardScaler()
    features_normalized = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

    n_timesteps = 10
    # Preprocess the data and store them as small arrays
    data_list = preprocess_data(features_normalized, labels, n_timesteps)

    # Extract the features and labels from the preprocessed data list
    X = np.array([item[0] for item in data_list])
    y = np.array([item[1] for item in data_list])
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test