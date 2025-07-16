# interpolation.py
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
# No direct use of choose_backend here, as dispatch is explicit via use_gpu

# Import GPU functions from your dedicated GPU implementation file
from .interpolation_gpu import gpu_linear_interpolation, gpu_cubic_spline_interpolation

def linear_interpolation(x, y, x_new, use_gpu=False):
    '''
    Perform linear interpolation using either CPU (SciPy) or GPU (CuPy) implementation.

    Arguments:
        x (numpy.ndarray): Given x values (must be strictly increasing).
        y (numpy.ndarray): Given y values (function values).
        x_new (numpy.ndarray): New x values to interpolate.
        use_gpu (bool): Whether to attempt GPU acceleration. Defaults to False.

    Returns:
        numpy.ndarray: Interpolated y values (always a NumPy array).
    '''
    if use_gpu:
        try:
            # Attempt GPU interpolation. The GPU function returns a NumPy array directly.
            return gpu_linear_interpolation(x, y, x_new)
        except Exception as e:
            # If GPU fails (e.g., CuPy not installed, memory error), print message and fall back to CPU.
            print(f"GPU linear interpolation failed, falling back to CPU: {e}")
            pass # Fall through to the CPU implementation below

    # CPU implementation using SciPy's interp1d
    interp_func = interp1d(x, y, kind='linear', fill_value="extrapolate")
    y_new = interp_func(x_new)
    return y_new

def spline_interpolation(x, y, x_new, bc_type='natural', use_gpu=False):
    """
    Perform cubic spline interpolation using either CPU (SciPy) or GPU (CuPy) implementation.

    Args:
        x (numpy.ndarray): Known x values (must be strictly increasing).
        y (numpy.ndarray): Known y values.
        x_new (numpy.ndarray): New x values where interpolation is needed.
        bc_type (str): Boundary condition type for CPU (SciPy) spline.
                       Common types: 'natural', 'clamped', 'not-a-knot'.
                       Note: The current GPU spline implementation uses SciPy's default
                       CubicSpline, which is 'not-a-knot', for coefficient calculation.
                       This `bc_type` primarily affects the CPU path.
        use_gpu (bool): Whether to attempt GPU acceleration. Defaults to False.

    Returns:
        numpy.ndarray: Interpolated y values at x_new points (always a NumPy array).
    """
    if use_gpu:
        try:
            # Attempt GPU cubic spline interpolation. The GPU function returns a NumPy array.
            return gpu_cubic_spline_interpolation(x, y, x_new)
        except Exception as e:
            # If GPU fails, print message and fall back to CPU.
            print(f"GPU cubic spline interpolation failed, falling back to CPU: {e}")
            pass # Fall through to the CPU implementation below

    # CPU implementation using SciPy's CubicSpline
    # Create cubic spline interpolator with the specified boundary condition.
    cs = CubicSpline(x, y, bc_type=bc_type)
    # Evaluate the spline at the new points.
    y_new = cs(x_new)
    return y_new