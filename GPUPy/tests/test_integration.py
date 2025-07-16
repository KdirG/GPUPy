# test_integration.py
import numpy as np
import pytest
from GPUPy.src.numerical_methods.integration import trapezoidal_integral, analytical_integral

# control the GPU availability
try:
    import cupy as cp
    gpu_available = True
except ImportError:
    cp = None
    gpu_available = False

# Test function and expected integral value(∫₀^π sin(x) dx = 2)
def f(x): return np.sin(x)
a, b = 0, np.pi
expected = 2.0

### ----------------- CPU TESTS -----------------

def test_trapezoidal_integral_cpu():
    x = np.linspace(a, b, 1000)
    y = np.sin(x)
    result = trapezoidal_integral(x, y, use_gpu=False)
    assert np.isclose(result, expected, atol=1e-3)

def test_analytical_integral_cpu():
    result, _ = analytical_integral(f, a, b, use_gpu=False)
    assert np.isclose(result, expected, atol=1e-8)

### ----------------- GPU TESTS-----------------

@pytest.mark.skipif(not gpu_available, reason="CuPy not available")
def test_trapezoidal_integral_gpu():
    x = cp.linspace(a, b, 1000)
    y = cp.sin(x)
    result = trapezoidal_integral(x, y, use_gpu=True)
    assert np.isclose(result, expected, atol=1e-3)

@pytest.mark.skipif(not gpu_available, reason="CuPy not available")
def test_analytical_integral_gpu():
    def f_gpu(x): return cp.sin(x)
    result, _ = analytical_integral(f_gpu, a, b, use_gpu=True)
    assert np.isclose(result, expected, atol=1e-3)
