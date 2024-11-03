'''
Linear Regression Using Normal Equation (easy)
Write a Python function that performs linear regression using the normal equation.
The function should take a matrix X (features) and a vector y (target) as input,
and return the coefficients of the linear regression model. Round your answer to four decimal
places, -0.0 is a valid result for rounding a very small number.
Example:
        input: X = [[1, 1], [1, 2], [1, 3]], y = [1, 2, 3]
        output: [0.0, 1.0]
        reasoning: The linear model is y = 0.0 + 1.0*x, perfectly fitting the input data.
'''

import numpy as np

def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    X = np.array(X)
    y = np.array(y).reshape(-1) ## reshaping into column vector
    X_trans = np.transpose(X)
    X_prod = np.linalg.inv(np.matmul(X_trans, X))
    y_prod = np.matmul(X_trans, y)
    theta = np.matmul(X_prod, y_prod) ## theta = ((X^T*X)^-1)*(X^T*y)
    theta = np.round(theta, 4).flatten().tolist()
    return theta

print(linear_regression_normal_equation([[1, 1], [1, 2], [1, 3]], [1, 2, 3]))
## [-0.0, 1.0]
print(linear_regression_normal_equation([[1, 3, 4], [1, 2, 5], [1, 3, 2]], [1, 2, 1]))
## [4.0, -1.0, -0.0]