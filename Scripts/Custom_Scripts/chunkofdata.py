import pandas as pd
from Custom_Scripts.TrainTestSplit import custom_train_test_split

def give_me_data_chunk(path):
    """
    Reads data from a CSV file, performs a custom train-test split, and returns the test data.

    Parameters:
    - path (str): The file path to the CSV file containing the dataset.

    Returns:
    - X_test (DataFrame): Features of the test set.
    - y_test (Series): Labels of the test set.

    Example:
    ```
    path = "path/to/dataset.csv"
    X_test, y_test = give_me_data_chunk(path)
    ```
    """
    # Read data from the CSV file
    data = pd.read_csv(path)

    # Perform custom train-test split
    X_train, X_test, y_train, y_test = custom_train_test_split(data)

    return X_test, y_test
