'''
Write a Python function that performs feature scaling on a dataset using both standardization
and min-max normalization. The function should take a 2D NumPy array as input,
where each row represents a data sample and each column represents a feature.
It should return two 2D NumPy arrays: one scaled by standardization and one by
min-max normalization. Make sure all results are rounded to the nearest 4th decimal.

Example:
        input: data = np.array([[1, 2], [3, 4], [5, 6]])
        output: ([[-1.2247, -1.2247], [0.0, 0.0], [1.2247, 1.2247]],
        [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        reasoning: Standardization rescales the feature to have a mean of 0 and
        a standard deviation of 1.
        Min-max normalization rescales the feature to a range of [0, 1],
        where the minimum feature value
        maps to 0 and the maximum to 1.
'''

import numpy as np

def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
    # Your code here
    def standard_scale(X):
        data_mean = np.mean(X, axis=0)
        std_dev = np.std(X, axis=0)
        std_scale = (X - data_mean) / (std_dev)
        return std_scale

    def min_max_scale(X):
        max_x = np.max(X, axis=0)
        min_x = np.min(X, axis=0)
        min_max_scale = (X - min_x) / (max_x - min_x)
        return min_max_scale

    standardized_data = standard_scale(data)
    normalized_data = min_max_scale(data)
    return np.round(standardized_data, 4).tolist(), np.round(normalized_data, 4).tolist()

print(feature_scaling(np.array([[1, 2], [3, 4], [5, 6]])))
# ([[-1.2247, -1.2247], [0.0, 0.0], [1.2247, 1.2247]], [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])