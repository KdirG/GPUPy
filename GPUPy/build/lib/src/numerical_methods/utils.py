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
def relative_error(approx, exact): #defining and returning relative error #TEKRAR BAK 0/0 GİBİ DURUMLAR İÇİN
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
def benchmark(method_func, *args, repeats=5, **kwargs): #method=> function to measure *args=>positional argument which we send to the function example=>sum(arr) arr in there repeats=>how many measurements we need **kwargs=>keyword arguments which we send to the function
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

    




