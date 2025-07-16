# test_linear_systems.py
import numpy as np
import pytest
from GPUPy.src.numerical_methods.linear_systems import solve_linear_system, solve_linear_system_lu

# Try to import CuPy
try:
    import cupy as cp
except ImportError:
    cp = None

# Test data
A = np.array([[3, 2], [1, 4]])
b = np.array([5, 6])
expected_x = np.linalg.solve(A, b)

### ----------------- CPU TESTS -----------------

def test_solve_linear_system_cpu():
    result = solve_linear_system(A, b, use_gpu=False)
    print(f"[CPU] Direct Solve Result: {result}")
    assert np.allclose(result, expected_x, atol=1e-6)

def test_solve_linear_system_lu_cpu():
    result = solve_linear_system_lu(A, b, use_gpu=False)
    print(f"[CPU] LU Solve Result: {result}")
    assert np.allclose(result, expected_x, atol=1e-6)

### ----------------- GPU TESTS -----------------

@pytest.mark.skipif(cp is None, reason="CuPy not available")
def test_solve_linear_system_gpu():
    result = solve_linear_system(A, b, use_gpu=True)
    print(f"[GPU] Direct Solve Result: {result}")
    assert np.allclose(result, expected_x, atol=1e-6)

@pytest.mark.skipif(cp is None, reason="CuPy not available")
def test_solve_linear_system_lu_gpu():
    result = solve_linear_system_lu(A, b, use_gpu=True)
    print(f"[GPU] LU Solve Result: {result}")
    assert np.allclose(result, expected_x, atol=1e-6)

