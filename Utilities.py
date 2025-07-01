import numpy as np

def generate_random_pattern(length):
    return np.array([1,-1,1,-1,1,-1,1,-1,1,-1])#np.random.choice([1, -1], size=length)

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
    
    return np.array([len(set(row) & unique_enco_ens) for row in off_ens])