'''
Write a Python function that takes the dot product of a matrix and a vector.
return -1 if the matrix could not be dotted with the vector
Example:
        input: a = [[1,2],[2,4]], b = [1,2]
        output:[5, 10]
        reasoning: 1*1 + 2*2 = 5;
                   1*2+ 2*4 = 10
'''
import numpy as np

def matrix_dot_vector(a: list[list[int]], b: list[int]) -> list[int]:

    arr1 = np.array(a)
    arr2 = np.array(b).reshape(-1, 1)
    m1, n1 = np.shape(arr1)
    m2, n2 = np.shape(arr2)
    if (n1 == m2):
        c = np.dot(arr1, arr2).flatten().tolist()
    else:
        c = -1
    return c

print(matrix_dot_vector([[1,2,3],[2,4,5],[6,8,9]],[1,2,3]))
# [14, 25, 49]
print(matrix_dot_vector([[1,2],[2,4],[6,8],[12,4]],[1,2,3]))
# -1