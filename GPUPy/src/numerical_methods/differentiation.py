# differentiation.py
import numpy as np
from .gpu_support import gradient_gpu

def compute_derivative(data, dx=1.0, method='auto', use_gpu=False):
    """
    Compute derivative of input data using different methods.
    
    Parameters:
        data: array-like
        dx: step size (default: 1.0)
        method: 'auto', 'forward', 'backward', 'central'
        use_gpu: True to use GPU-accelerated version if available
    Returns:
        Approximate derivative
    """
    if use_gpu:  # GPU integration
        return gradient_gpu(data, dx)
    if method == 'auto':
        return np.gradient(data, dx)
    elif method == 'forward':
        return forward_diff(data, dx)
    elif method == 'backward':
        return backward_diff(data, dx)
    elif method == 'central':
        return central_diff(data, dx)
    else:
        raise ValueError("Invalid method. Choose 'auto', 'forward', 'backward', or 'central'.")

def forward_diff(data, dx=1.0):
    """Forward difference method for array data."""
    result = np.zeros_like(data)
    result[:-1] = (data[1:] - data[:-1]) / dx
    result[-1] = result[-2]  # Handle boundary
    return result

def backward_diff(data, dx=1.0):
    """Backward difference method for array data."""
    result = np.zeros_like(data)
    result[1:] = (data[1:] - data[:-1]) / dx
    result[0] = result[1]  # Handle boundary
    return result

def central_diff(data, dx=1.0):
    """Central difference method for array data."""
    result = np.zeros_like(data)
    result[1:-1] = (data[2:] - data[:-2]) / (2 * dx)
    result[0] = (data[1] - data[0]) / dx  # Forward diff for first point
    result[-1] = (data[-1] - data[-2]) / dx  # Backward diff for last point
    return result



    
    
    