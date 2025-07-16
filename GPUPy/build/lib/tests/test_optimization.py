# test_optimization.py
import pytest
import numpy as np
import cupy as cp
from .optimization import minimize_scalar_wrapper, minimize_wrapper



# =============================
# Scalar Optimization - CPU
# =============================
def test_minimize_scalar_cpu():
    def f(x): return (x - 3)**2
    result = minimize_scalar_wrapper(f, use_gpu=False)
    assert result.success
    assert abs(result.x - 3.0) < 1e-4

# =============================
# Multivariate Optimization - CPU
# =============================
def test_minimize_multivariate_cpu():
    def f(x): return (x[0] - 2)**2 + (x[1] + 1)**2
    result = minimize_wrapper(f, x0=np.array([0.0, 0.0]), use_gpu=False)
    assert result.success
    assert np.allclose(result.x, [2.0, -1.0], atol=1e-4)    
    
@pytest.mark.skipif(cp is None, reason="CuPy not available")
def test_minimize_scalar_gpu():
    def f(x): return (x - 3)**2
    result = minimize_scalar_wrapper(f, use_gpu=True)
    print(f"[GPU Scalar] x: {result.x}, success: {result.success}")
    assert result.success
    assert abs(result.x - 3.0) < 1e-4

@pytest.mark.skipif(cp is None, reason="CuPy not available")
def test_minimize_multivariate_gpu():
    def f(x): return (x[0] - 2)**2 + (x[1] + 1)**2
    result = minimize_wrapper(f, x0=[0.0, 0.0], use_gpu=True)
    print(f"[GPU Multi] x: {result.x}, success: {result.success}")
    assert result.success
    assert np.allclose(result.x, [2.0, -1.0], atol=1e-4)
    