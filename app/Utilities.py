import json
import numpy as np

def generate_random_pattern(length):
    return np.random.choice([1, -1], size=length)

def center_of_mass_rowwise(matrix):
    matrix = np.asarray(matrix)
    cols = np.arange(matrix.shape[1])
    com = (matrix * cols).sum(axis=1) / matrix.sum(axis=1)
    return com

def get_active_neurons(frs, th=1):
    return np.where(frs>th)[0]

def rowwise_correlation(mat1, mat2):
    """
    Computes Pearson correlation coefficient between corresponding rows of two matrices.
    
    Parameters:
        mat1, mat2 (2D np.array): Input matrices of shape (n_rows, n_cols)
        
    Returns:
        correlations (1D np.array): Correlation for each row pair
    """
    mat1 = np.asarray(mat1)
    mat2 = np.asarray(mat2)
    
    assert mat1.shape == mat2.shape, "Input matrices must have the same shape"
    
    # Subtract mean along each row
    mat1_mean_centered = mat1 - mat1.mean(axis=1, keepdims=True)
    mat2_mean_centered = mat2 - mat2.mean(axis=1, keepdims=True)
    
    # Compute dot product of centered rows
    numerator = np.sum(mat1_mean_centered * mat2_mean_centered, axis=1)
    
    # Compute norms
    denominator = np.linalg.norm(mat1_mean_centered, axis=1) * np.linalg.norm(mat2_mean_centered, axis=1)
    
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        correlation = numerator / denominator
        correlation = np.nan_to_num(correlation)  # convert NaNs to 0 if needed
    
    return correlation

def center_of_mass_columnwise_blocked(matrix, block_size=10):
    matrix = np.asarray(matrix)
    n_cols = matrix.shape[1]
    n_blocks = n_cols // block_size

    means = []
    stds = []

    for i in range(n_blocks):
        block = matrix[:, i * block_size:(i + 1) * block_size]
        rows = np.arange(matrix.shape[0])
        col_sums = block.sum(axis=0)

        with np.errstate(divide='ignore', invalid='ignore'):
            com = (block.T * rows).sum(axis=1) / col_sums
            com = np.where(col_sums == 0, np.nan, com)

        # Drop NaNs before calculating stats
        valid_com = com[~np.isnan(com)]
        means.append(np.mean(valid_com) if valid_com.size > 0 else np.nan)
        stds.append(np.std(valid_com)/block_size if valid_com.size > 0 else np.nan)

    return np.array(means), np.array(stds)

def ensamble_overlap(enco_ens, off_ens):
    # all_values = np.concatenate(enco_ens)
    unique_enco_ens = set(np.unique(enco_ens))
    
    return np.array([set(row) & unique_enco_ens for row in off_ens])


def remove_top_percent_columns(matrix, percent):
    """
    Remove the columns corresponding to the top `percent` of values
    in the first row of the matrix.

    Parameters:
        matrix (2D array-like): Input matrix.
        percent (float): Percentage of top columns to remove (0–100).

    Returns:
        filtered_matrix (ndarray): Matrix with selected columns removed.
        removed_indices (ndarray): Indices of the removed columns.
    """
    matrix = np.array(matrix)
    n_cols = matrix.shape[1]
    n_remove = max(1, int(np.ceil((percent / 100) * n_cols)))

    # Get indices of top `percent` values in the first row
    top_indices = np.argsort(matrix[0])[::-1][:n_remove]

    # Remove those columns
    result = np.delete(matrix, top_indices, axis=1)

    return result, top_indices

def remove_inactive_cells(matrix,theta):
    matrix = np.array(matrix)
    zero_activity_neurons = np.all(matrix < theta, axis=1)
    
    removed_indices = np.where(zero_activity_neurons)[0]
    filtered_matrix = matrix[~zero_activity_neurons]
    
    return filtered_matrix, removed_indices

def day_wise_avg_offline_activity(matrix, block_size=10):
    matrix = np.array(matrix)
    n_rep, n_neurons = matrix.shape

    # Ensure the number of rows is divisible by block_size
    if n_rep % block_size != 0:
        raise ValueError(f"Number of rows ({n_rep}) must be divisible by block_size ({block_size}).")

    # Reshape and compute mean across blocks of rows
    reshaped = matrix.reshape(-1, block_size, n_neurons)  # shape: (num_blocks, block_size, cols)
    block_means = reshaped.mean(axis=1)  # mean over rows in each block

    return block_means

def save_params(params, json_output_path):
    """
    dumps top-level assignments as JSON.
    """
    # Dump to JSON
    with open(json_output_path, 'w+') as out:
        json.dump(params, out, indent=4)
    print("Parameters saved to: ", json_output_path)

def top_percent_indices(arr, percent):
    """
    Returns the indices of the top `percent`% largest elements in the array.
    
    Parameters:
        arr (array-like): Input array.
        percent (float): Percentage (0–100) of top elements to select.
    
    Returns:
        np.ndarray: Indices of top elements in descending order of value.
    """
    arr = np.asarray(arr)
    n = len(arr)
    
    if percent <= 0 or percent > 100:
        raise ValueError("percent must be in the range (0, 100]")

    n_top = max(1, int(np.ceil((percent / 100) * n)))  # at least 1 element
    sorted_indices = np.argsort(arr)[::-1]  # descending order
    top_indices = sorted_indices[:n_top]
    
    return top_indices

import numpy as np

def average_firing_rates_with_active(firing_rates, T_FC, T_offline,T_ir, Nday, Nrep, T_recall,ID, threshold=0.0):
    """
    Compute average firing rates above a threshold and list of active neurons
    for T_FC, each day, and T_recall.
    
    Parameters:
    -----------
    firing_rates : np.ndarray
        Matrix of shape (N, T) with firing rates of N neurons over T time steps.
    T_FC : int
        Number of time steps for first phase (T_FC).
    T_offline : int
        Number of time steps for each offline repetition.
    Nday : int
        Number of days.
    Nrep : int
        Number of repetitions per day in offline phase.
    T_recall : int
        Number of time steps for recall phase.
    threshold : float
        Only consider firing rates above this threshold when computing averages.
        
    Returns:
    --------
    avg_FC : np.ndarray
        Average firing rate of each neuron during T_FC (above threshold). Shape (N,)
    active_FC : np.ndarray
        Boolean array indicating which neurons were active during T_FC. Shape (N,)
    avg_days : np.ndarray
        Average firing rate of each neuron for each day (above threshold). Shape (N, Nday)
    active_days : np.ndarray
        Boolean array indicating active neurons per day. Shape (N, Nday)
    avg_recall : np.ndarray
        Average firing rate of each neuron during recall (above threshold). Shape (N,)
    active_recall : np.ndarray
        Boolean array indicating which neurons were active during recall. Shape (N,)
    """
    N, T_total = firing_rates.shape
    
    # Check consistency
    expected_T = T_FC + ID + Nday * (Nrep * (T_offline + T_ir) + ID) + T_recall + ID
    if T_total != expected_T:
        raise ValueError(f"Total time steps mismatch: expected {expected_T}, got {T_total}")
    
    # Helper function: mean above threshold
    def mean_above_threshold(data, thresh):
        masked = data[data > thresh]
        if len(masked) == 0:
            return 0.0
        return masked.mean()
    
    # Average and active neurons for T_FC
    avg_FC = np.array([mean_above_threshold(firing_rates[i, :T_FC], threshold) for i in range(N)])
    active_FC = np.any(firing_rates[:, :T_FC] > threshold, axis=1)
    active_neurons_FC = np.where(active_FC)[0]  
    # Average and active neurons for each day
    avg_days = np.zeros((N, Nday,Nrep), dtype=float)
    active_days = np.zeros((N, Nday,Nrep), dtype=bool)
    for day in range(Nday):
        for rep in range(Nrep):
            start_idx = T_FC + ID + day * (Nrep * (T_offline + T_ir) + ID) + rep * (T_offline + T_ir)
            end_idx = start_idx + T_offline
            avg_days[:, day,rep] = np.array([mean_above_threshold(firing_rates[i, start_idx:end_idx], threshold) for i in range(N)])
            active_days[:, day,rep] = np.any(firing_rates[:, start_idx:end_idx] > threshold, axis=1)
    active_neurons_days = [np.where(active_days[:, day])[0] for day in range(Nday)]
    # Average and active neurons for recall
    avg_recall = np.array([mean_above_threshold(firing_rates[i, -(T_recall+ID):-(ID)], threshold) for i in range(N)])
    active_recall = np.any(firing_rates[:, -(T_recall+ID):-(ID)] > threshold, axis=1)
    active_neurons_recall = np.where(active_recall)[0]
    return avg_FC, active_neurons_FC, avg_days, active_neurons_days, avg_recall, active_neurons_recall

