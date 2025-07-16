# test_differentiation.py
import numpy as np
import pytest
from GPUPy.src.numerical_methods.differentiation import (
    compute_derivative,
    forward_diff,
    backward_diff,
    central_diff
)

# We will test the GPU function separately
try:
    from GPUPy.src.numerical_methods.gpu_support import gradient_gpu
    import cupy as cp
except ImportError:
    gradient_gpu = None
    cp = None

# Test data
x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)
expected_dy = np.cos(x)  # derivative: cos(x)

### ----------------- CPU TESTS -----------------

def test_forward_diff():
    dy = forward_diff(y, dx=x[1] - x[0])
    assert np.allclose(dy, expected_dy, atol=0.1)

def test_backward_diff():
    dy = backward_diff(y, dx=x[1] - x[0])
    assert np.allclose(dy, expected_dy, atol=0.1)

def test_central_diff():
    dy = central_diff(y, dx=x[1] - x[0])
    assert np.allclose(dy, expected_dy, atol=0.01)  # Daha hassas

def test_compute_derivative_auto():
    dy = compute_derivative(y, dx=x[1] - x[0], method='auto')
    assert np.allclose(dy, expected_dy, atol=0.01)

def test_compute_derivative_manual_methods():
    for method in ['forward', 'backward', 'central']:
        dy = compute_derivative(y, dx=x[1] - x[0], method=method)
        assert np.allclose(dy, expected_dy, atol=0.1)

### ----------------- GPU TEST -----------------

@pytest.mark.skipif(gradient_gpu is None or cp is None, reason="CuPy not available")
def test_gpu_derivative():
    y_gpu = cp.asarray(y)
    dy_gpu = gradient_gpu(y_gpu, dx=x[1] - x[0])
    dy_cpu = cp.asnumpy(dy_gpu)
    assert np.allclose(dy_cpu, expected_dy, atol=0.01)
