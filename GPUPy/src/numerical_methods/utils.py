# utils.py
import time
import cupy as cp
import numpy as np

def choose_backend(use_gpu=None):
    """
    Choose the backend for computation (CPU or GPU).
    
    Args:
        use_gpu: Boolean to specify whether to use GPU (True) or CPU (False).
    
    Returns:
        Backend module (NumPy for CPU or CuPy for GPU).
    """
    if use_gpu is None:
        return np  # Default to CPU if no choice is provided
    elif use_gpu:
        try:
            return cp  # Use cupy (GPU) if specified
        except ImportError:
            print("Warning: CuPy not available. Falling back to NumPy (CPU).")
            return np
    else:
        return np  # Use numpy (CPU) if specified

#measuring time
def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()                      # Taking the start time
        result = func(*args, **kwargs)           # Calling the prime function
        end = time.time()                        # Taking the finish time
        print(f"{func.__name__} took {end - start:.6f}s")  # Printing the time
        return result                            # Returning the result back
    return  wrapper #when we need wrapping this function  to the another function example=>time(home())
    
#error calculation    
def relative_error(approx, exact): #defining and returning relative error 
    try:
        return abs((approx - exact) / exact)
    except: 
        ZeroDivisionError
    return float('inf')

def absolute_error(approx, exact): #defining and returning absolute error 
    return abs(approx - exact)

#convergence check
def has_converged(old_val, new_val, tol=1e-6):
    # Eğer değerler GPU (CuPy) üzerindeyse
    if isinstance(old_val, cp.ndarray) or isinstance(new_val, cp.ndarray):
        return float(cp.abs(new_val - old_val)) < tol
    else:
        # CPU (NumPy veya temel Python) değerleri için
        return float(abs(new_val - old_val)) < tol

#benchmark supporter => for gpu vs cpu comparation
def benchmark(method_func, *args, repeats=5, **kwargs): 
      # Initial call to validate (not timed)
    # We do this outside the loop to handle any initial validation
    method_func(*args, **kwargs)
    
    # Now time the method calls
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        method(*args, **kwargs)
        durations.append(time.perf_counter() - start)
    
    avg_time = sum(durations) / repeats
    return avg_time

def custom_benchmark(method, func, a, b, repeats=5, **kwargs): #custom benchmark for high polynominals root finding
    """Custom benchmark that verifies sign change before each call."""
    # Verify sign change
    fa = func(a)
    fb = func(b)
    if fa * fb >= 0:
        raise ValueError("Function values at interval endpoints must have opposite signs.")
        
        
#numpy-cupy converter=> when we need a convertion for an array between numpy and cupy
def to_gpu_array(arr): 
    try:
        import cupy as cp #looking for are we have cupy library
        return cp.array(arr) #numpy to cupy
    except ImportError: #if cupy library isnt imported 
        return arr #return converted cupy array

def to_cpu_array(arr):
    try:
        return arr.get()  # cupy to numpy
    except AttributeError: #if we dont have a cupy array which we need to convert
        return arr #return converted numpy array



#converting functions to a String
def compile_function_from_string(func_str, var='x'): #
    return lambda x: eval(func_str, {var: x})

def create_ode_func_from_string(func_str, use_gpu_for_func_creation=False):
    
    xp = cp if (_CUPY_AVAILABLE_IN_UTILS and use_gpu_for_func_creation) else np

    # We are using `exec` to dynamically create a function.
    # This is more powerful than `_safe_eval_expression` but potentially more risky.
    # For production environments, such functions are not recommended due to security concerns.
    # However, for the structure of the current Gradio application, this approach is adopted.
    try:
        # Construct the function body. It will accept `y` and `t` arguments.
        # The user's provided `func_str` will be used directly as the `return` statement.
        func_code = f"def _ode_func(y, t):\n    return {func_str}"

        # Prepare the namespace in which this function will operate.
        # This should be similar to `safe_namespace` in `_safe_eval_expression`.
        exec_namespace = {
            "__builtins__": {
                "abs": abs, "min": min, "max": max, "sum": sum, "round": round,
                "len": len, "list": list, "dict": dict, "tuple": tuple, "set": set,
                "str": str, "int": int, "float": float, "bool": bool,
                "__import__": None # Crucially, disable __import__!
            },
            'np': np,
            'math': np,
        }
        if _CUPY_AVAILABLE_IN_UTILS and xp is cp:
            exec_namespace['cp'] = cp
            exec_namespace['xp'] = cp # Also add 'xp' alias
        else:
            exec_namespace['xp'] = np # Add 'xp' alias as numpy

        # Use `exec` to define the `_ode_func` function within this namespace.
        # The function becomes part of `exec_namespace`.
        exec(func_code, exec_namespace)
        _f = exec_namespace['_ode_func'] # Retrieve the newly created function

        # Now, return a wrapper function to comply with the ODE solver's expected interface
        # (a function that takes y, t). This wrapper will convert inputs to the appropriate backend
        # and then call `_f`.
        def wrapped_ode_function(y_val, t_val):
            # Convert the input `y_val` to the selected backend (NumPy or CuPy).
            # This might already be handled within odeint_wrapper, but we add it here
            # to make the function itself flexible.
            if xp is cp:
                if isinstance(y_val, np.ndarray):
                    y_converted = cp.asarray(y_val)
                else: # If already a CuPy array or scalar
                    y_converted = y_val
            else: # xp is np
                if isinstance(y_val, cp.ndarray):
                    y_converted = y_val.get() # Convert from CuPy to NumPy
                else: # If already a NumPy array or scalar
                    y_converted = y_val

            result = _f(y_converted, t_val)
            
            # Convert the result back to NumPy for Gradio or other CPU-based operations.
            if isinstance(result, cp.ndarray) and xp is cp:
                return result.get() # Convert from CuPy to NumPy
            return result

        return wrapped_ode_function

    except (SyntaxError, NameError, TypeError, ValueError) as e:
        raise ValueError(f"ODE function expression could not be compiled: {e}. Expression: '{func_str}'")
    except Exception as e:
        raise Exception(f"Unexpected error while compiling ODE function: {e}. Expression: '{func_str}'")



