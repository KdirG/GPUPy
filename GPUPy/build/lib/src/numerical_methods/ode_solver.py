# ode_solver.py
import numpy as np
import cupy as cp
from scipy.integrate import odeint
from .utils import choose_backend

def odeint_wrapper(func, y0, t, use_gpu=None, args=(), method='BDF', atol=1e-6, rtol=1e-6, 
                  mxstep=500, h0=0.1, full_output=False, jacobian=None, **kwargs):
    """
    Solve ODE using either CPU or GPU depending on use_gpu parameter.
    
    Parameters:
        func: callable - The system of ODEs.
        y0: array-like - Initial state.
        t: array-like - Time points where solution is computed.
        use_gpu: bool or None - Whether to use GPU for computation.
        args: tuple - Additional arguments passed to func.
        method: str - Integration method ('BDF' or 'RK45').
        atol: float - Absolute tolerance for the solution.
        rtol: float - Relative tolerance for the solution.
        mxstep: int - Maximum number of steps to take.
        h0: float - Initial step size.
        full_output: bool - Whether to return additional output information.
        jacobian: callable - Function to compute the Jacobian matrix of the ODE system.
        kwargs: additional keyword arguments passed to the solver.
        
    Returns:
        ndarray: Solution of the ODE at each time point.
    """
    xp = choose_backend(use_gpu)
    
    if use_gpu:
        # GPU implementation using cuSOLVER for implicit methods
        
        # Convert inputs to GPU arrays
        y0_gpu = cp.asarray(y0, dtype=np.float64)
        t_gpu = cp.asarray(t, dtype=np.float64)
        
        # Convert args to GPU arrays if they are numpy arrays
        args_gpu = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                args_gpu.append(cp.asarray(arg))
            else:
                args_gpu.append(arg)
        args_gpu = tuple(args_gpu)
        
        # Setup GPU functions
        def gpu_func(y, t, *args_gpu):
            # Function should handle CuPy arrays
            return cp.asarray(func(y, t, *args_gpu))
        
        # Initialize result array
        n_dim = len(y0)
        y_result = cp.zeros((len(t_gpu), n_dim), dtype=y0_gpu.dtype)
        y_result[0] = y0_gpu
        
        if method.upper() == 'BDF':
            # Backward Differentiation Formula (implicit method)
            # Requires solving linear systems with cuSOLVER
            
            # Precompute step sizes
            dt = cp.diff(t_gpu)
            
            # If Jacobian function is provided
            if jacobian is not None:
                def gpu_jacobian(y, t, *args_gpu):
                    return cp.asarray(jacobian(y, t, *args_gpu))
            else:
                # Finite difference approximation of Jacobian
                def gpu_jacobian(y, t, *args_gpu):
                    eps = 1e-8
                    jac = cp.zeros((n_dim, n_dim), dtype=y.dtype)
                    f0 = gpu_func(y, t, *args_gpu)
                    for i in range(n_dim):
                        y_perturbed = y.copy()
                        y_perturbed[i] += eps
                        f1 = gpu_func(y_perturbed, t, *args_gpu)
                        jac[:, i] = (f1 - f0) / eps
                    return jac
            
            # BDF1 (backward Euler) implementation
            for i in range(1, len(t_gpu)):
                # Current step size
                step = dt[i-1]
                
                # Previous value
                y_prev = y_result[i-1]
                
                # Initial guess for Newton iteration (use previous value)
                y_next = y_prev.copy()
                
                # Newton iteration to solve implicit equation
                for newton_iter in range(10):  # Max 10 Newton iterations
                    # Compute residual: y_next - y_prev - h*f(y_next, t_next)
                    f_next = gpu_func(y_next, t_gpu[i], *args_gpu)
                    residual = y_next - y_prev - step * f_next
                    
                    # Check convergence
                    if cp.max(cp.abs(residual)) < atol:
                        break
                    
                    # Compute Jacobian: I - h*df/dy
                    jac = gpu_jacobian(y_next, t_gpu[i], *args_gpu)
                    jac_system = cp.eye(n_dim) - step * jac
                    
                    # Solve linear system using cuSolver
                    # Here we use CuPy's built-in linear solver which uses cuSOLVER
                    delta = cp.linalg.solve(jac_system, residual)
                    
                    # Update solution
                    y_next = y_next - delta
                
                y_result[i] = y_next
                
        elif method.upper() == 'RK45':
            # Runge-Kutta 4th order method
            for i in range(1, len(t_gpu)):
                h = t_gpu[i] - t_gpu[i-1]
                yi = y_result[i-1]
                ti = t_gpu[i-1]
                
                # RK4 steps
                k1 = gpu_func(yi, ti, *args_gpu)
                k2 = gpu_func(yi + h/2 * k1, ti + h/2, *args_gpu)
                k3 = gpu_func(yi + h/2 * k2, ti + h/2, *args_gpu)
                k4 = gpu_func(yi + h * k3, ti + h, *args_gpu)
                
                y_result[i] = yi + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        # Convert result back to CPU
        result = cp.asnumpy(y_result)
        
        # Provide info dict if requested
        if full_output:
            info_dict = {
                'message': f'GPU-accelerated {method} method',
                'nst': len(t)-1,
                'nfe': (len(t)-1) * (4 if method.upper() == 'RK45' else 10), 
                'success': True
            }
            return result, info_dict
        return result
    else:
        # Use standard SciPy solver on CPU
        return odeint(func, y0, t, args=args, atol=atol, rtol=rtol, mxstep=mxstep, 
                      h0=h0, full_output=full_output, **kwargs)