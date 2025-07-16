# core.py
#function for matrix multiplication
import numpy as np
def matrix_multiply(A,B):
 A = np.array(A)
 B = np.array(B)
 if A.shape[1] != B.shape[0]:
    raise ValueError("Matrix dimensions are incompatible: The number of columns of A must be equal to the number of rows of B.")

 return np.dot(A, B)
 