# integration.py

import numpy as np
from scipy.integrate import trapezoid, quad
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
    xp = choose_backend(use_gpu) # because we use choose_backend, if use_gpu=False return np

    # if CuPy is used move data to the GPU
    if xp.__name__ == 'cupy' and use_gpu:
        x_arr = xp.asarray(x)
        y_arr = xp.asarray(y)
        dx = x_arr[1:] - x_arr[:-1]
        y_sum = y_arr[:-1] + y_arr[1:]
        integral = xp.sum(dx * y_sum) / 2.0
        return float(integral.get()) # give CuPy result to the CPU
    else:
        # if there is no numpy or cupy do fallback
        return trapezoid(y, x) # SciPy trapezoid used directly, y and x must be numpy arrays anyway


def analytical_integral(func, a, b, use_gpu=False, num_points=1000):
    # If use_gpu, try to use CuPy
    if use_gpu:
        try:
            import cupy as cp
            # print("INFO: Attempting GPU analytical integral with CuPy.") # For debugging
            x = cp.linspace(a, b, num_points)
            y = func(x) # func should process cupy arrays

            integral_val = trapezoidal_integral(x.get(), y.get(), use_gpu=True) # sending numpy array to trapezoidal_integral
                                                                              # or trapezoidal_integral should accept cupy arrays
                                                                              # since func(x) returns a cupy array, get() is needed to convert to numpy
                                                                              # or trapezoidal_integral should correctly process cupy arrays.
                                                                              # Current trapezoidal_integral does cp.asarray(x), so get() is not strictly necessary

            # If calculation was done with CuPy, integral_val should already be a float.
            # Error estimation, a simple formula for the trapezoidal rule
            # This is only an approximate error estimate, not as accurate as quad.
            error_estimate = abs(integral_val) * ((b - a) / num_points)**2 / 12

            return integral_val, error_estimate # return 2 values

        except ImportError:
            print("CuPy not available. Falling back to CPU for analytical integral.")
            # Fall back to CPU if CuPy is not present
            return quad(func, a, b) # quad already returns (integral, error)
        except Exception as e:
            print(f"Error during CuPy analytical integral: {e}. Falling back to CPU.")
            # Fall back to CPU if another error occurs with CuPy
            return quad(func, a, b) # quad already returns (integral, error)
    else:
        # If use_gpu=False, directly use quad on CPU
        # print("INFO: Performing CPU analytical integral with SciPy quad.") # For debugging
        return quad(func, a, b) # quad already returns (integral, error)


        
