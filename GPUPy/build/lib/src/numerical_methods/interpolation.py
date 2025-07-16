# interpolation.py
import numpy as np
from scipy.interpolate import CubicSpline, interp1d
import matplotlib.pyplot as plt
from .interpolation_gpu import gpu_linear_interpolation, gpu_cubic_spline_interpolation
from .utils import choose_backend

def linear_interpolation(x, y, x_new, use_gpu=None):
    '''
    Perform linear interpolation using either CPU or GPU implementation.
    
    Arguments:
        x (array): Given x values (indices)
        y (array): Given y values (function values) 
        x_new (array): New x values to interpolate 
        use_gpu (bool): Whether to use GPU acceleration (default: False)
    
    Returns:
        array: Interpolated y values
    '''
    if use_gpu:
        try:
            return gpu_linear_interpolation(x, y, x_new)
        except Exception as e:
            print(f"GPU interpolation failed, falling back to CPU: {e}")
            use_gpu = False
    
    if not use_gpu:
        interp_func = interp1d(x, y, kind='linear', fill_value="extrapolate")
        y_new = interp_func(x_new)
        return y_new

def spline_interpolation(x, y, x_new, bc_type, use_gpu=None):
    """
    Perform cubic spline interpolation using SciPy's CubicSpline.
    
    Args:
        x (array): Known x values (must be strictly increasing)
        y (array): Known y values
        x_new (array): New x values where interpolation is needed
        bc_type (str): Boundary condition type:
                      - 'natural': natural spline (second derivative = 0 at boundaries)
                      - 'clamped': first derivative specified at boundaries
                      - 'not-a-knot': continuous third derivative at first/last interior points
    
    Returns:
        array: Interpolated y values at x_new points
    """
    if use_gpu:
        try:
            return gpu_cubic_spline_interpolation(x, y, x_new)
        except Exception as e:
            print(f"GPU interpolation failed, falling back to CPU: {e}")
            use_gpu = False
    
    if not use_gpu:
        # Create cubic spline interpolator
        cs = CubicSpline(x, y, bc_type=bc_type)
        # Evaluate at new points
        y_new = cs(x_new)
        return y_new

     
     
    
