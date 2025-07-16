#linear_systems.py
import numpy as np
import cupy as cp
from GPUPy.src.numerical_methods.utils import choose_backend
from scipy.linalg import lu_factor, lu_solve
import cupyx.scipy.linalg as cpx_linalg


def solve_linear_system(A, b, use_gpu=None):
    """
    Solve a linear system Ax = b
    
    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        use_gpu: Boolean flag to indicate whether to use GPU (default: None)
    
    Returns:
        x: Solution vector
    """
    # Choose the appropriate backend
    xp = choose_backend(use_gpu)
    
    # Convert inputs to the selected backend's array format
    A_arr = xp.asarray(A)
    b_arr = xp.asarray(b)
    
    # Solve the system using the selected backend
    x_arr = xp.linalg.solve(A_arr, b_arr)
    
    # If using GPU, convert result back to CPU NumPy array
    if use_gpu:
        return cp.asnumpy(x_arr)
    else:
        return x_arr

def solve_linear_system_lu(A, b, use_gpu=None):
    """
    Solve a linear system Ax = b using LU decomposition.
    
    Args:
        A: Coefficient matrix
        b: Right-hand side vector
        use_gpu: Boolean flag to indicate whether to use GPU (default: None)
        
    Returns:
        x: Solution vector
    """
    if use_gpu:
        # Convert inputs to CuPy arrays
        A_gpu = cp.asarray(A)
        b_gpu = cp.asarray(b)

        # Use CuPy's LU decomposition (SciPy-compatible)
        lu, piv = cpx_linalg.lu_factor(A_gpu)
        x_gpu = cpx_linalg.lu_solve((lu, piv), b_gpu)

        return cp.asnumpy(x_gpu)  # convert back to NumPy
    else:
        # CPU solution
        A_cpu = np.asarray(A)
        b_cpu = np.asarray(b)
        lu, piv = lu_factor(A_cpu)
        x = lu_solve((lu, piv), b_cpu)
        return x

