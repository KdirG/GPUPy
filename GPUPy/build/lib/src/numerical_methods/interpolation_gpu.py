import numpy as np
import cupy as cp

def gpu_linear_interpolation(x, y, x_new):
    """
    Linear interpolation function running on GPU
    """
    # Move input data to GPU
    x_gpu = cp.asarray(x)
    y_gpu = cp.asarray(y)
    x_new_gpu = cp.asarray(x_new)
    
    # Create array to store results
    y_new = cp.zeros_like(x_new_gpu)
    
    # Perform interpolation for each x_new point
    for i, x_point in enumerate(x_new_gpu):
        # Find the nearest points for linear interpolation
        mask = x_gpu <= x_point
        if not cp.any(mask):
            i0 = 0
        else:
            i0 = cp.where(mask)[0][-1]
        
        if i0 >= len(x_gpu) - 1:
            y_new[i] = y_gpu[-1]
            continue
        
        i1 = i0 + 1
        
        # Linear interpolation formula
        x0, x1 = x_gpu[i0], x_gpu[i1]
        y0, y1 = y_gpu[i0], y_gpu[i1]
        
        y_new[i] = y0 + (x_point - x0) * (y1 - y0) / (x1 - x0)
    
    # Return result
    return y_new

def gpu_cubic_spline_interpolation(x, y, x_new):
    """
    Cubic spline interpolation function running on GPU
    """
    # Prepare data on CPU
    from scipy.interpolate import CubicSpline
    cs = CubicSpline(x, y)
    
    # Get coefficients
    c0 = cs.c[0]
    c1 = cs.c[1]
    c2 = cs.c[2]
    c3 = cs.c[3]
    knots = cs.x
    
    # Move coefficients to GPU
    c0_gpu = cp.asarray(c0)
    c1_gpu = cp.asarray(c1)
    c2_gpu = cp.asarray(c2)
    c3_gpu = cp.asarray(c3)
    knots_gpu = cp.asarray(knots)
    x_new_gpu = cp.asarray(x_new)
    
    # Create array to store results
    y_new = cp.zeros_like(x_new_gpu)
    
    # Compute spline value for each x_new point
    for i, x_point in enumerate(x_new_gpu):
        # Find the interval in which x_point lies
        mask = knots_gpu <= x_point
        if not cp.any(mask):
            idx = 0
        else:
            idx = cp.where(mask)[0][-1]
        
        if idx >= len(knots_gpu) - 1:
            idx = len(knots_gpu) - 2
        
        # Normalized x value
        dx = x_point - knots_gpu[idx]
        
        # Evaluate cubic polynomial
        y_new[i] = c3_gpu[idx] + dx * (c2_gpu[idx] + dx * (c1_gpu[idx] + dx * c0_gpu[idx]))
    
    return y_new

# More efficient alternative (for linear interpolation)
def gpu_linear_interpolation_vectorized(x, y, x_new):
    """
    More efficient linear interpolation function running on GPU
    Uses vectorized operations for faster performance
    """
    # Move input data to GPU
    x_gpu = cp.asarray(x)
    y_gpu = cp.asarray(y)
    x_new_gpu = cp.asarray(x_new)
    
    # For each x_new value, find its position in x_gpu
    indices = cp.zeros(len(x_new_gpu), dtype=int)
    
    for i, x_val in enumerate(x_new_gpu):
        mask = x_gpu <= x_val
        if not cp.any(mask):
            indices[i] = 0
        else:
            indices[i] = cp.where(mask)[0][-1]
    
    # Boundary check
    valid_mask = indices < len(x_gpu) - 1
    
    # Compute interpolation for valid index values
    i0 = indices[valid_mask]
    i1 = i0 + 1
    
    x0 = x_gpu[i0]
    x1 = x_gpu[i1]
    y0 = y_gpu[i0]
    y1 = y_gpu[i1]
    
    # Select correct x_new values
    x_points = x_new_gpu[valid_mask]
    
    # Create result array
    y_new = cp.zeros_like(x_new_gpu)
    
    # Handle cases where i0 = len(x_gpu) - 1
    edge_mask = ~valid_mask
    if cp.any(edge_mask):
        y_new[edge_mask] = y_gpu[-1]
    
    # Perform interpolation
    y_new[valid_mask] = y0 + (x_points - x0) * (y1 - y0) / (x1 - x0)
    
    return y_new
