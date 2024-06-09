import matplotlib.pyplot as plt
import seaborn as sns

def visualize_preprocessed_data(path, X_train, title):

    """
    Visualize preprocessed data features for a sample.

    Args:
    - X_train (array-like): Preprocessed input features.
    - y_train (array-like): Preprocessed target labels.
    - title (str): Title for the plot.

    Returns:
    - None

    Description:
    This function visualizes the preprocessed features for a selected sample.
    It creates a heatmap to display the features across different time steps.

    Steps:
    1. Choose a sample index for visualization (default is 0).
    2. Plot the features using a heatmap, where each cell represents a feature value at a specific time step.
    3. Annotate the heatmap with the feature values.
    4. Set the title, x-axis label (Feature Index), and y-axis label (Time Step).
    5. Show the plot.

    Note:
    - The input data X_train should be a 3D array, where the dimensions represent (samples, time steps, features).
        -- Refer preprocessing code to get an idea of how the data is preprocessed for this code.
    """

    # Choose a sample index for visualization
    sample_index = 0
    # Plot the features for the selected sample
    plt.figure(figsize=(10, 6))
    sns.heatmap(X_train[sample_index, :, :], cmap='viridis', annot=True, fmt=".2f")
    plt.title(f'{title}: Preprocessed Features for Sample Index {sample_index}')
    plt.xlabel('Feature Index')
    plt.ylabel('Time Step')
    plt.show()