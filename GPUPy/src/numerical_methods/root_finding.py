# root_finding.py
import numpy as np
import cupy as cp
from .utils import choose_backend
from .utils import has_converged

def bisection(func, a, b, tolerance=1e-6, max_iterations=100, use_gpu=None):
    """
    Implements the bisection method to find a root of a function.
    
    Args:
        func: The function for which to find the root (f(x)).
        a: The lower bound of the initial interval.
        b: The upper bound of the initial interval.
        tolerance: The desired accuracy (stopping criterion based on interval width).
        max_iterations: Maximum number of iterations to prevent infinite loops.
        use_gpu: Boolean to specify whether to use GPU (True) or CPU (False).
        
    Returns:
        The approximate root if found, or None if not found within max_iterations.
    """
    # Choose backend based on use_gpu parameter
    xp = choose_backend(use_gpu)
    
    # Convert inputs to arrays of the selected backend
    a = xp.array(a, dtype=xp.float64)
    b = xp.array(b, dtype=xp.float64)
    
    # Check if the interval contains a root
    fa = func(a)
    fb = func(b)
    if fa * fb >= 0:
        raise ValueError("Function values at interval endpoints must have opposite signs.")
    
    iteration_count = 0
    while (b - a) / 2.0 > tolerance and iteration_count < max_iterations:
        c = (a + b) / 2.0  # Calculate midpoint
        fc = func(c)
        
        if xp.abs(fc) < tolerance:
            return c  # Root found within tolerance
        elif fa * fc < 0:
            b = c      # Root is in [a, c]
            fb = fc
        else:
            a = c      # Root is in [c, b]
            fa = fc
            
        iteration_count += 1
    
    if iteration_count == max_iterations:
        print(f"Warning: Bisection method reached maximum iterations ({max_iterations}). Convergence may not be achieved within tolerance.")
    
    # Return midpoint as approximate root
    result = (a + b) / 2.0
    
    # If using GPU, convert result back to CPU if it's a scalar
    if use_gpu and isinstance(result, cp.ndarray) and result.size == 1:
        return float(result)
    return result

def newton_raphson(f, df, x0, tol=1e-6, max_iter=100, use_gpu=None):
    """
    Newton-Raphson method for finding a root of a function using its derivative.
    
    Args:
        f: Function whose root is to be found.
        df: Derivative of the function.
        x0: Initial guess.
        tol: Tolerance for convergence.
        max_iter: Maximum number of iterations.
        use_gpu: Boolean to specify whether to use GPU (True) or CPU (False).
        
    Returns:
        Approximate root value.
        
    Raises:
        ValueError: If derivative is zero or method fails to converge.
    """
    # Choose backend based on use_gpu parameter
    xp = choose_backend(use_gpu)
    
    # Convert initial guess to array of the selected backend
    x = xp.array(x0, dtype=xp.float64)
    
    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)
        
        if xp.abs(dfx) < 1e-12:  # Small derivative control
            raise ValueError("Derivative too close to zero; division by zero risk.")
        
        x_new = x - fx / dfx
        
        if has_converged(x, x_new, tol):
            # If using GPU, convert result back to CPU if it's a scalar
            if use_gpu and isinstance(x_new, cp.ndarray) and x_new.size == 1:
                return float(x_new)
            return x_new
            
        x = x_new
    
    raise ValueError(f"Newton-Raphson did not converge after {max_iter} iterations.")

'''
# Example usage
if __name__ == "__main__":
    # Define functions compatible with both NumPy and CuPy
    def example_func(x):
        # This function must work with both NumPy and CuPy arrays
        return x**2 - 4
    
    def example_deriv(x):
        # Derivative of x^2 - 4
        return 2*x
    
    # Try with CPU
    try:
        root_cpu = bisection(example_func, 0, 3, use_gpu=False)
        print(f"Root found with CPU bisection: {root_cpu}")
        
        root_newton_cpu = newton_raphson(example_func, example_deriv, 3.0, use_gpu=False)
        print(f"Root found with CPU Newton-Raphson: {root_newton_cpu}")
    except Exception as e:
        print(f"CPU computation failed: {e}")
    
    # Try with GPU
    try:
        root_gpu = bisection(example_func, 0, 3, use_gpu=True)
        print(f"Root found with GPU bisection: {root_gpu}")
        
        root_newton_gpu = newton_raphson(example_func, example_deriv, 3.0, use_gpu=True)
        print(f"Root found with GPU Newton-Raphson: {root_newton_gpu}")
    except Exception as e:
        print(f"GPU computation failed: {e}")
'''