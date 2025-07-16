# optimization.py
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from .utils import choose_backend

# Import CuPy if available for GPU support
try:
    import cupy as cp
except ImportError:
    cp = None

def minimize_scalar_wrapper(func, use_gpu=None, method='brent', bounds=None, options=None):
    """
    Minimize a scalar function with optional GPU acceleration.
    
    Args:
        func: Function to minimize
        use_gpu: Whether to use GPU for function evaluation
        method: Optimization method ('brent', 'bounded', 'golden', etc.)
        bounds: Bounds for the search interval
        options: Additional options for the optimizer
    
    Returns:
        res: Optimization result
    """
    if use_gpu and cp is not None:
        # Create a wrapper function that moves data to GPU, evaluates, and returns to CPU
        def gpu_func(x):
            x_gpu = cp.asarray(x)
            result_gpu = func(x_gpu)  # Function should handle GPU arrays
            return float(cp.asnumpy(result_gpu))  # Convert back to scalar CPU value
        
        # Use SciPy's CPU optimizer but with GPU-accelerated function evaluations
        result = minimize_scalar(gpu_func, method=method, bounds=bounds, options=options)
    else:
        # Standard CPU optimization
        result = minimize_scalar(func, method=method, bounds=bounds, options=options)
    
    return result
    
def minimize_wrapper(func, x0, use_gpu=None, method='BFGS', jac=None, bounds=None, constraints=None, options=None):
    """
    Minimize a multivariate function with optional GPU acceleration.
    
    Args:
        func: Function to minimize
        x0: Initial guess
        use_gpu: Whether to use GPU for function evaluation
        method: Optimization method ('BFGS', 'L-BFGS-B', 'SLSQP', etc.)
        jac: Jacobian (gradient) function
        bounds: Bounds for the variables
        constraints: Optimization constraints
        options: Additional options for the optimizer
    
    Returns:
        res: Optimization result
    """
    if use_gpu and cp is not None:
        # Create a wrapper function that moves data to GPU, evaluates, and returns to CPU
        def gpu_func(x):
            x_gpu = cp.asarray(x)
            result_gpu = func(x_gpu)  # Function should handle GPU arrays
            return cp.asnumpy(result_gpu)  # Convert back to CPU array
        
        # Handle Jacobian if provided
        if jac is not None:
            def gpu_jac(x):
                x_gpu = cp.asarray(x)
                result_gpu = jac(x_gpu)
                return cp.asnumpy(result_gpu)
        else:
            gpu_jac = None
            
        result = minimize(gpu_func, x0, method=method, jac=gpu_jac, 
                         bounds=bounds, constraints=constraints, options=options)
    else:
        # Standard CPU optimization
        result = minimize(func, x0, method=method, jac=jac, 
                         bounds=bounds, constraints=constraints, options=options)
    
    return result