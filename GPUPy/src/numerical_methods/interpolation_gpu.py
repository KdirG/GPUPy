import numpy as np
import cupy as cp
# Import CubicSpline here as well, since it's used directly in this file
from scipy.interpolate import CubicSpline

def gpu_linear_interpolation(x, y, x_new):
    # Move input data to GPU
    x_gpu = cp.asarray(x)
    y_gpu = cp.asarray(y)
    x_new_gpu = cp.asarray(x_new)

    # For each x_new value, find its position in x_gpu.
    # cp.searchsorted returns insertion points, which correspond to the right boundary
    # of the interval where x_new_gpu[i] would be inserted. Subtracting 1 gives the index
    # of the left boundary of the interval.
    indices = cp.searchsorted(x_gpu, x_new_gpu, side='right') - 1

    # Clip indices to valid range for interpolation [0, len(x_gpu) - 2].
    # This handles extrapolation for points outside the original x range by clamping
    # them to the first or last interval.
    indices = cp.clip(indices, 0, len(x_gpu) - 2)

    # Get the bounding points for interpolation: (x0, y0) and (x1, y1)
    i0 = indices
    i1 = indices + 1

    x0 = x_gpu[i0]
    x1 = x_gpu[i1]
    y0 = y_gpu[i0]
    y1 = y_gpu[i1]

    # Perform linear interpolation using the formula: y = y0 + (x_new - x0) * (y1 - y0) / (x1 - x0)
    y_new_gpu = y0 + (x_new_gpu - x0) * (y1 - y0) / (x1 - x0)

    # Handle extrapolation explicitly for points exactly outside the original x range.
    # For x_new values smaller than x_gpu[0], set y_new to y_gpu[0].
    y_new_gpu[x_new_gpu < x_gpu[0]] = y_gpu[0]
    # For x_new values larger than x_gpu[-1], set y_new to y_gpu[-1].
    y_new_gpu[x_new_gpu > x_gpu[-1]] = y_gpu[-1]

    # Return result as a NumPy array (explicit conversion from CuPy to NumPy)
    return cp.asnumpy(y_new_gpu)

def gpu_cubic_spline_interpolation(x, y, x_new):
    # Prepare data on CPU to get spline coefficients using SciPy's CubicSpline.
    # x and y are assumed to be NumPy arrays.
    # CubicSpline.c stores coefficients as cs.c[j, i] for (x-x[i])**(k-j).
    # For cubic spline (k=3), j=0,1,2,3 correspond to powers 3,2,1,0.
    cs = CubicSpline(x, y)

    # Extract coefficients for each interval.
    # c3_coeffs_cpu: coefficients for (x-x_i)^3
    # c2_coeffs_cpu: coefficients for (x-x_i)^2
    # c1_coeffs_cpu: coefficients for (x-x_i)^1
    # c0_coeffs_cpu: coefficients for (x-x_i)^0 (constant term)
    c3_coeffs_cpu = cs.c[0]
    c2_coeffs_cpu = cs.c[1]
    c1_coeffs_cpu = cs.c[2]
    c0_coeffs_cpu = cs.c[3]
    knots_cpu = cs.x # The knot points (original x values)

    # Move coefficients and new x values to GPU.
    c0_gpu = cp.asarray(c0_coeffs_cpu)
    c1_gpu = cp.asarray(c1_coeffs_cpu)
    c2_gpu = cp.asarray(c2_coeffs_cpu)
    c3_gpu = cp.asarray(c3_coeffs_cpu)
    knots_gpu = cp.asarray(knots_cpu)
    x_new_gpu = cp.asarray(x_new)

    # Find the interval for each x_new point.
    # `indices` will hold the index `i` such that `knots_gpu[i] <= x_new_gpu < knots_gpu[i+1]`.
    indices = cp.searchsorted(knots_gpu, x_new_gpu, side='right') - 1

    # Clip indices to ensure they are within the valid range for accessing coefficients.
    # The valid range for `indices` is `[0, len(knots_gpu) - 2]`.
    # This handles extrapolation by clamping to the first or last interval's polynomial.
    indices = cp.clip(indices, 0, len(knots_gpu) - 2)

    # Calculate `dx = x_new - knots[indices]` for each `x_new` point.
    # This `dx` is `(x - x_i)` in the polynomial `c3*(x-x_i)^3 + ... + c0`.
    dx_gpu = x_new_gpu - knots_gpu[indices]

    # Evaluate cubic polynomial using Horner's method (vectorized on GPU).
    # P(dx) = c0 + c1*dx + c2*dx^2 + c3*dx^3
    # This is equivalent to: P(dx) = c0 + dx * (c1 + dx * (c2 + dx * c3))
    y_new_gpu = c0_gpu[indices] + dx_gpu * (c1_gpu[indices] + dx_gpu * (c2_gpu[indices] + dx_gpu * c3_gpu[indices]))

    # Return result as a NumPy array (explicit conversion from CuPy to NumPy).
    return cp.asnumpy(y_new_gpu)