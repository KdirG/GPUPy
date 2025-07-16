# test_interpolation.py
import numpy as np
import pytest
from .interpolation import linear_interpolation, spline_interpolation
from .gpu_interpolation import gpu_linear_interpolation_vectorized  # Assuming it's defined here

# Check if GPU is available
try:
    import cupy as cp
    gpu_available = True
except ImportError:
    cp = None
    gpu_available = False

# Test data
x = np.array([0, 1, 2, 3, 4])
y = np.array([0, 1, 4, 9, 16])
x_new = np.array([1.5, 2.5, 3.5])  # New x values for interpolation
expected = np.array([2.25, 6.25, 12.25])  # Expected results
bc_type = 'natural'  # Boundary condition for spline

### ----------------- CPU TEST -----------------

def test_linear_interpolation_cpu():
    result = linear_interpolation(x, y, x_new, use_gpu=False)
    assert np.allclose(result, expected, atol=1e-3)

def test_spline_interpolation_cpu():
    result = spline_interpolation(x, y, x_new, bc_type, use_gpu=False)
    assert np.allclose(result, expected, atol=1e-3)

### ----------------- GPU TEST -----------------

@pytest.mark.skipif(not gpu_available, reason="CuPy not available")
def test_linear_interpolation_gpu():
    result = linear_interpolation(x, y, x_new, use_gpu=True)
    assert np.allclose(result, expected, atol=1e-3)

@pytest.mark.skipif(not gpu_available, reason="CuPy not available")
def test_spline_interpolation_gpu():
    result = spline_interpolation(x, y, x_new, bc_type, use_gpu=True)
    assert np.allclose(result, expected, atol=1e-3)

# ----------------- CPU TEST -----------------
def test_gpu_linear_interpolation_vectorized_cpu():
    # Perform vectorized linear interpolation on CPU using NumPy
    result = gpu_linear_interpolation_vectorized(x, y, x_new)
    # Compare with expected values
    assert np.allclose(result.get(), expected, atol=1e-3)  # Use .get() to bring GPU result to CPU

# ----------------- GPU TEST -----------------
@pytest.mark.skipif(not gpu_available, reason="CuPy not available")
def test_gpu_linear_interpolation_vectorized_gpu():
    # Perform vectorized linear interpolation on GPU using CuPy
    x_gpu = cp.asarray(x)
    y_gpu = cp.asarray(y)
    x_new_gpu = cp.asarray(x_new)

    result_gpu = gpu_linear_interpolation_vectorized(x_gpu, y_gpu, x_new_gpu)
    
    # Compare with expected values on GPU
    assert cp.allclose(result_gpu, expected, atol=1e-3)

