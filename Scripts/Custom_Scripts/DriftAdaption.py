from Custom_Scripts.model_architecture import model_initialisation

def AdaptDrift(X_train, X_test, y_train, y_test, model = None):

    """
    Adapt the model to drift by training on new data.

    Parameters:
    - X_train (numpy.ndarray): Training features.
    - X_test (numpy.ndarray): Test features.
    - y_train (numpy.ndarray): Training labels.
    - y_test (numpy.ndarray): Test labels.

    Returns:
    - keras.models.Model: Adapted model trained on the new data.
    """    

    model = model_initialisation()
    history = model.fit(X_train, y_train, 
                        epochs=50, 
                        batch_size=1000, 
                        validation_data=(X_test, y_test),  # Use validation_data for X_test and y_test
                        verbose=1)
    return model
