'''
Linear Regression Using Gradient Descent (easy)
Write a Python function that performs linear regression using gradient descent.
The function should take NumPy arrays X (features with a column of ones for the intercept)
and y (target) as input, along with learning rate alpha and the number of iterations,
and return the coefficients of the linear regression model as a NumPy array.
Round your answer to four decimal places. -0.0 is a valid result for rounding a
very small number.
'''

import numpy as np

def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
    # Your code here, make sure to round
    m, n = X.shape
    theta = np.zeros((n, 1)) # create a column vector of zeros with height n and width 1
    y_reshaped = y.reshape(-1, 1) # reshaping y into a 2D array
    X_trans = np.transpose(X)

    for _ in range(iterations):
        predictions = np.matmul(X, theta)
        error = predictions - y_reshaped
        updates = np.matmul(X_trans, error) # computed gradient of cost function
        updated_m = updates/m
        theta -= alpha * updated_m # algorithm for gradient descent
    return np.round(theta.flatten(), 4)

print(linear_regression_gradient_descent(np.array([[1, 1], [1, 2], [1, 3]]), np.array([1, 2, 3]), 0.01, 1000))
## [0.1107, 0.9513]