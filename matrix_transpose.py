'''
Transpose of a Matrix (easy)
Write a Python function that computes the transpose of a given matrix.
Example
Example:
        input: a = [[1,2,3],[4,5,6]]
        output: [[1,4],[2,5],[3,6]]
        reasoning: The transpose of a matrix is obtained by flipping rows and columns.
'''

import numpy as np

def transpose_matrix(a: list[list[int]]) -> list[list[int]]:
    b = np.array(a)
    b = np.transpose(b)
    b = b.tolist()
    return b

print(transpose_matrix([[1, 2], [3, 4], [5, 6]]))
## [[1, 3, 5], [2, 4, 6]]
print(transpose_matrix([[1, 2, 3], [4, 5, 6]]))
# [[1, 4], [2, 5], [3, 6]]
