import tensorflow as tf
from tensorflow import keras
import numpy as np

def model_initialisation():

    """
    Initialize a convolutional neural network (CNN) model for federated learning.

    Returns:
    - model (keras.Model): Initialized CNN model.

    Description:
    This function initializes a CNN model with specified architecture and configurations suitable for federated learning
    tasks. The model architecture consists of convolutional layers with average pooling, batch normalization, dropout,
    and dense layers. The model is compiled with categorical crossentropy loss and Adam optimizer with learning rate
    scheduling.

    Model Architecture:
    - Input Shape: (10, 17) - Represents 10 time steps with 17 features each.
    - Convolutional Layers:
      - 1st Conv1D Layer: Filters=32, Kernel Size=7, Activation=ReLU, Padding=Same.
      - Batch Normalization Layer.
      - Average Pooling Layer: Pool Size=2.
      - 2nd Conv1D Layer: Filters=16, Kernel Size=7, Activation=ReLU, Padding=Same.
      - Batch Normalization Layer.
      - Dropout Layer: Rate=0.5.
      - Average Pooling Layer: Pool Size=2.
    - Flatten Layer: Flattens the output for dense layers.
    - RepeatVector Layer: Repeats the output for each time step.
    - TimeDistributed Dense Layer: Units=num_classes (5), Activation=Softmax.

    Optimizer and Learning Rate Schedule:
    - Learning Rate Schedule: Exponential Decay with initial learning rate=0.01, decay steps=10000, decay rate=0.9.
    - Optimizer: Adam with learning rate scheduling.

    Returns the initialized CNN model for federated learning.

    Note:
    - The function relies on TensorFlow and Keras libraries for model initialization and compilation.
    - The actual implementation of the model architecture, optimizer, and learning rate scheduling is provided
      within this documentation.
    """
    num_classes = 5
    input_shape = (10, 17)
    tf.keras.backend.clear_session()
    tf.random.set_seed(100)
    np.random.seed(100)

    model = keras.models.Sequential()

    # Add Convolutional layers with AveragePooling
    model.add(keras.layers.Conv1D(filters=32, kernel_size=7, padding='same', activation='relu', input_shape=input_shape))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.AveragePooling1D(pool_size=2))

    model.add(keras.layers.Conv1D(filters=16, kernel_size=7, padding='same', activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.AveragePooling1D(pool_size=2))

    # Flatten the output for Dense layers
    model.add(keras.layers.Flatten())

    # Repeat the Dense layer for each time step
    model.add(keras.layers.RepeatVector(input_shape[0]))

    # Add another Dense layer for each time step
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=num_classes, activation='softmax')))

    # Learning Rate Schedule
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.01, decay_steps=10000, decay_rate=0.9)
    
    # Using Adam optimizer with learning rate scheduling
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])

    return model