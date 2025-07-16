# test_ode_solver.py
import pytest
import numpy as np
from .ode_solver import odeint_wrapper  # Correct import of the Ode solver function

# Simple ODE example: dy/dt = -y
def simple_ode(y, t):
    return -y

# Test 1: Solving a simple ODE
def test_odeint_wrapper_simple_ode():
    # Initial condition and time array
    y0 = [1.0]
    t = np.linspace(0, 5, 100)
    
    # When the use_gpu parameter is set to False, the CPU solution will be used
    result = odeint_wrapper(simple_ode, y0, t, use_gpu=False)
    
    # Test: Check that the solution's shape is correct
    assert result.shape == (len(t), len(y0)), "Solution shape is incorrect."
    
    # Test: Check that the ODE solution decreases over time (since dy/dt = -y, y should decrease over time)
    assert result[-1] < result[0], "ODE solution is not decreasing as expected."

# Test 2: Check the accuracy of the solution
def test_odeint_wrapper_accuracy():
    y0 = [1.0]
    t = np.linspace(0, 5, 100)
    
    # Analytical solution: y(t) = exp(-t)
    analytical_solution = np.exp(-t)
    
    # When the use_gpu parameter is set to False, the CPU solution will be used
    result = odeint_wrapper(simple_ode, y0, t, use_gpu=False)
    
    # Test: Check that the difference between the analytical and numerical solution is within an acceptable tolerance
    tolerance = 1e-4
    assert np.allclose(result[:, 0], analytical_solution, atol=tolerance), f"Solution does not match analytical solution within {tolerance} tolerance."

# Test 3: Check that the response is as expected
def test_odeint_wrapper_response():
    y0 = [2.0]
    t = np.linspace(0, 5, 100)
    
    # Expected ODE solution: dy/dt = -y, initial condition y0 = 2.0
    expected_response = np.exp(-t) * 2.0
    
    # When the use_gpu parameter is set to False, the CPU solution will be used
    result = odeint_wrapper(simple_ode, y0, t, use_gpu=False)
    
    # Test: Check that the solution matches the expected response
    assert np.allclose(result[:, 0], expected_response, atol=1e-4), "Response is different from expected."

# Test 4: Test GPU usage
@pytest.mark.parametrize("use_gpu", [True, False])
def test_odeint_wrapper_gpu_usage(use_gpu):
    y0 = [1.0]
    t = np.linspace(0, 5, 100)
    
    result = odeint_wrapper(simple_ode, y0, t, use_gpu=use_gpu)
    
    # Test: Check that the solution's shape is correct
    assert result.shape == (len(t), len(y0)), "Solution shape is incorrect."

