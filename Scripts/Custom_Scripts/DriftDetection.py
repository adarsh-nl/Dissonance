import numpy as np
from scipy.stats import beta
import scipy.stats as stats
import json
import os

from Custom_Scripts.constants import BASE_PATH
from Custom_Scripts.debug import debug
DEBUG = False

def drift_detection(client_id, comm_round, Q, delta=20, sensitivity=0.05):

    """
    Perform drift detection on the given data.

    Parameters:
    - client_id (int): The ID of the client.
    - comm_round (int): The communication round.
    - Q (numpy.ndarray): An array of data points to detect drift.
    - delta (int, optional): The parameter for drift detection. Defaults to 20.
    - sensitivity (float, optional): Sensitivity parameter. Defaults to 0.05.

    Returns:
    - bool: True if drift is detected, False otherwise.
    """

    Status  = False
    # Save the results to a JSON file
    directory_path = BASE_PATH + f'Results\client{client_id}\DriftDetection\\'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    output_file_path = directory_path + f'CommunicationRound_{comm_round}.json'

    sf = 0
    n = len(Q)
    lambda_val = sensitivity
    Th = -np.log(lambda_val)

    results = {
                    "client_id": client_id,
                    "comm_round": comm_round,
                    "Delta": delta,
                    "Sensitivity": sensitivity,
                    "Lambda_val":lambda_val,
                    "Th": Th,
                    "Recent_scores_mean" : None,
                    "Old_scores_mean" : None,
                    "alpha_a": None,
                    "beta_a": None,
                    "alpha_b": None,
                    "beta_b": None,
                    "sk": None,
                    "sf": sf
                }
    
    for k in range(delta, n - delta):
        Qa = Q[:k]
        Qb = Q[k:]
        results["Old_scores_mean"] = float(np.mean(Qa))
        results["Recent_scores_mean"] = float(np.mean(Qb))
        if np.mean(Qa) <= (1 - lambda_val) * np.mean(Qb):
            alpha_a, beta_a = estimate_beta_parameters(Qa)
            alpha_b, beta_b = estimate_beta_parameters(Qb)
            results["alpha_a"] = alpha_a
            results["alpha_b"] = alpha_b
            results["beta_a"] = beta_a
            results["beta_b"] = beta_b

            sk = calculate_log_likelihood_ratio(Q, alpha_a, beta_a, alpha_b, beta_b, delta, n)
            results["sk"] = sk
            sf = max(sf, sk)
            results["sf"] = sf

            # Detect change if sk is greater than the threshold
            if sf > Th:
                Status = True

    with open(output_file_path, 'w') as file:
        json.dump(results, file, indent=2)

    return Status

def estimate_beta_parameters(Q):

    """
    Estimate the shape parameters (alpha, beta) for a beta distribution.

    Parameters:
    - Q (numpy.ndarray): Data points for estimation.

    Returns:
    - tuple: Estimated alpha and beta parameters.
    """

    mean_q = np.mean(Q)
    var_q = np.var(Q)

    # Check if variance is zero
    if var_q == 0:
        # Set default values or handle the situation accordingly
        alpha_hat, beta_hat = 1.0, 1.0
    else:
        alpha_hat = (((1 - mean_q) / var_q) - (1 / mean_q)) * np.square(mean_q)
        beta_hat = alpha_hat * ((1 / mean_q) - 1)

    return alpha_hat, beta_hat


def beta_distribution_pdf(x, alpha, beta):

    """
    Calculate the probability density function (PDF) of a beta distribution.

    Parameters:
    - x (float): Value at which to evaluate the PDF.
    - alpha (float): Shape parameter.
    - beta (float): Shape parameter.

    Returns:
    - float: PDF value at the given x.
    """

    if alpha <= 0 or beta <= 0:
        raise ValueError("Shape parameters alpha and beta must be greater than 0.")
    
    pdf = stats.beta.pdf(x, alpha, beta)
    return pdf

def calculate_log_likelihood_ratio(Q, alpha_a, beta_a, alpha_b, beta_b, delta, n):

    """
    Calculate the log-likelihood ratio for drift detection.

    Parameters:
    - Q (numpy.ndarray): Data points for calculation.
    - alpha_a (float): Alpha parameter for the first distribution.
    - beta_a (float): Beta parameter for the first distribution.
    - alpha_b (float): Alpha parameter for the second distribution.
    - beta_b (float): Beta parameter for the second distribution.
    - delta (int): Drift detection parameter.
    - n (int): Length of the data.

    Returns:
    - float: Log-likelihood ratio value.
    """

    sk = 0
    for k in range(delta, n):
        a_likelihood = beta_distribution_pdf(Q[k], alpha_a, beta_a)
        b_likelihood = beta_distribution_pdf(Q[k], alpha_b, beta_b)

        # Check if b_likelihood is zero before performing the division
        if b_likelihood != 0:
            sk = sk + np.log(a_likelihood / b_likelihood)
        else:
            # Handle the case where b_likelihood is zero
            sk = sk + np.log(a_likelihood)

    return sk