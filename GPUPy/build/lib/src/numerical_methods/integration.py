# integration.py

import numpy as np
from scipy.integrate import trapezoid,quad
from .utils import choose_backend

def trapezoidal_integral(x, y, use_gpu=None):
    """
    Compute the integral using the trapezoidal rule.
    
    Parameters:
        x (array): Array of x values (must be monotonically increasing)
        y (array): Array of y values corresponding to x
        use_gpu (bool): Whether to use GPU calculation

    Returns:
        float: Approximate integral value
    """
    xp = choose_backend(use_gpu)
    x_arr = xp.asarray(x)
    y_arr = xp.asarray(y)
    
    # Manual implementation of trapezoidal rule for GPU
    dx = x_arr[1:] - x_arr[:-1]
    y_sum = y_arr[:-1] + y_arr[1:]
    integral = xp.sum(dx * y_sum) / 2.0
    
    # Convert result to CPU (scalar value)
    if hasattr(integral, 'get'):
        return float(integral.get())
    return float(integral)
    
    # Use SciPy's implementation for CPU
    return trapezoid(y_arr, x_arr)

def analytical_integral(func, a, b, use_gpu=False, num_points=1000):
    if use_gpu:
        try:
            import cupy as cp
            x = cp.linspace(a, b, num_points)
            
            # Direct evaluation: func must handle cp.ndarray
            y = func(x)

            integral = trapezoidal_integral(x, y, use_gpu=True)
            error_estimate = abs(integral) * ((b - a) / num_points)**2 / 12
            return integral, error_estimate

        except ImportError:
            print("CuPy not available. Falling back to CPU.")
            use_gpu = False

    if not use_gpu:
        from scipy.integrate import quad
        return quad(func, a, b)



        
