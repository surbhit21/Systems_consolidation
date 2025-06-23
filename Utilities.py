import numpy as np

def generate_random_pattern(length,seed = 2025):
    np.random.seed(seed)
    return np.random.choice([1, -1], size=length)