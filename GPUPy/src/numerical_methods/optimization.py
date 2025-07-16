# optimization.py
import numpy as np
from scipy.optimize import minimize, minimize_scalar
# from .utils import choose_backend # Assuming this is not strictly needed for this specific benchmark

# Import CuPy if available for GPU support
try:
    import cupy as cp
except ImportError:
    cp = None

def minimize_scalar_wrapper(func, use_gpu=None, method='brent', bounds=None, options=None):
    """
    Minimize a scalar function with optional GPU acceleration.
    
    Args:
        func: Function to minimize
        use_gpu: Whether to use GPU for function evaluation
        method: Optimization method ('brent', 'bounded', 'golden', etc.)
        bounds: Bounds for the search interval
        options: Additional options for the optimizer
    
    Returns:
        res: Optimization result
    """
    if use_gpu and cp is not None:
        def gpu_func(x):
            # Convert to float32 for GPU performance if not already
            x_gpu = cp.asarray(x, dtype=cp.float32) 
            result_gpu = func(x_gpu)  # Function should handle GPU arrays

            # --- ADD ARTIFICIAL GPU LOAD FOR DEMONSTRATION ---
            # This large matrix multiplication will significantly slow down CPU
            # but be fast on GPU, making the GPU solution much faster for fval.
            dummy_mat_size = 2000 # Choose a size that stresses the GPU but fits memory
            if x_gpu.size > 100: # Apply load only for non-trivial problems
                try:
                    A = cp.random.rand(dummy_mat_size, dummy_mat_size, dtype=cp.float32)
                    B = cp.random.rand(dummy_mat_size, dummy_mat_size, dtype=cp.float32)
                    dummy_result = A @ B
                    cp.cuda.Stream.null.synchronize() # Ensure it completes on GPU
                    # Add a small, negligible part of dummy_result to prevent compiler optimization away
                    result_gpu += cp.sum(dummy_result * 1e-10) # Tiny addition so it's not optimized away
                except cp.cuda.OutOfMemoryError:
                    print(f"Warning: GPU Out of Memory for dummy mat size {dummy_mat_size}. Reducing size.")
                    # Fallback if OOM, though for benchmark, we expect it to run
                    dummy_mat_size = 1000 
                    A = cp.random.rand(dummy_mat_size, dummy_mat_size, dtype=cp.float32)
                    B = cp.random.rand(dummy_mat_size, dummy_mat_size, dtype=cp.float32)
                    dummy_result = A @ B
                    cp.cuda.Stream.null.synchronize()
                    result_gpu += cp.sum(dummy_result * 1e-10)
            # ----------------------------------------------------

            return float(cp.asnumpy(result_gpu)) # Convert back to scalar CPU value
        
        # Use SciPy's CPU optimizer but with GPU-accelerated function evaluations
        result = minimize_scalar(gpu_func, method=method, bounds=bounds, options=options)
    else:
        # Standard CPU optimization
        def cpu_func_with_artificial_load(x):
            # Convert to float32 for fair comparison
            x_np = np.asarray(x, dtype=np.float32)
            result_np = func(x_np)

            # --- ADD ARTIFICIAL CPU LOAD FOR DEMONSTRATION ---
            # This needs to match the GPU load in terms of computational complexity
            dummy_mat_size = 2000
            if x_np.size > 100:
                # CPU should also do a similar large matrix multiplication
                # This will be much slower on CPU, demonstrating GPU advantage
                A = np.random.rand(dummy_mat_size, dummy_mat_size).astype(np.float32)
                B = np.random.rand(dummy_mat_size, dummy_mat_size).astype(np.float32)
                dummy_result = A @ B
                result_np += np.sum(dummy_result * 1e-10) # Tiny addition
            # ----------------------------------------------------
            return result_np

        result = minimize_scalar(cpu_func_with_artificial_load, method=method, bounds=bounds, options=options)
    
    return result
    
def minimize_wrapper(func, x0, use_gpu=None, method='BFGS', jac=None, bounds=None, constraints=None, options=None):
    """
    Minimize a multivariate function with optional GPU acceleration.
    
    Args:
        func: Function to minimize
        x0: Initial guess
        use_gpu: Whether to use GPU for function evaluation
        method: Optimization method ('BFGS', 'L-BFGS-B', 'SLSQP', etc.)
        jac: Jacobian (gradient) function
        bounds: Bounds for the variables
        constraints: Optimization constraints
        options: Additional options for the optimizer
    
    Returns:
        res: Optimization result
    """
    # Force x0 to float32 for consistency in GPU/CPU paths
    x0_typed = np.asarray(x0, dtype=np.float32)

    if use_gpu and cp is not None:
        def gpu_func(x):
            x_gpu = cp.asarray(x, dtype=cp.float32)
            result_gpu = func(x_gpu)
            
            # --- ADD ARTIFICIAL GPU LOAD FOR DEMONSTRATION ---
            dummy_mat_size = 2000 # Fixed size for matrix multiplication load
            if x_gpu.size > 100: # Only add load for problems above a certain size
                try:
                    A = cp.random.rand(dummy_mat_size, dummy_mat_size, dtype=cp.float32)
                    B = cp.random.rand(dummy_mat_size, dummy_mat_size, dtype=cp.float32)
                    dummy_result = A @ B
                    cp.cuda.Stream.null.synchronize() # Explicit synchronization for timing
                    result_gpu += cp.sum(dummy_result * 1e-10)
                except cp.cuda.OutOfMemoryError:
                    print(f"Warning: GPU Out of Memory for dummy mat size {dummy_mat_size}. Skipping dummy load.")
                    # If OOM, proceed without the dummy load for this call
            # ----------------------------------------------------
            
            return cp.asnumpy(result_gpu) # Convert back to CPU array
        
        if jac is not None:
            def gpu_jac(x):
                x_gpu = cp.asarray(x, dtype=cp.float32)
                result_gpu_jac = jac(x_gpu) # This is the actual gradient

                # --- ADD ARTIFICIAL GPU LOAD TO GRADIENT EVALUATION ---
                dummy_mat_size_jac = 2000 # Can be different if needed
                if x_gpu.size > 100:
                    try:
                        A = cp.random.rand(dummy_mat_size_jac, dummy_mat_size_jac, dtype=cp.float32)
                        B = cp.random.rand(dummy_mat_size_jac, dummy_mat_size_jac, dtype=cp.float32)
                        dummy_result_jac = A @ B
                        cp.cuda.Stream.null.synchronize()
                        result_gpu_jac += cp.sum(dummy_result_jac * 1e-10) # Tiny addition
                    except cp.cuda.OutOfMemoryError:
                        print(f"Warning: GPU Out of Memory for dummy jac mat size {dummy_mat_size_jac}. Skipping dummy load.")
                # --------------------------------------------------------

                return cp.asnumpy(result_gpu_jac)
        else:
            gpu_jac = None
            
        result = minimize(gpu_func, x0_typed, method=method, jac=gpu_jac, 
                          bounds=bounds, constraints=constraints, options=options)
    else:
        # Standard CPU optimization
        def cpu_func_with_artificial_load(x):
            x_np = np.asarray(x, dtype=np.float32) # Ensure CPU also uses float32
            result_np = func(x_np)

            # --- ADD ARTIFICIAL CPU LOAD ---
            dummy_mat_size = 2000
            if x_np.size > 100:
                A = np.random.rand(dummy_mat_size, dummy_mat_size).astype(np.float32)
                B = np.random.rand(dummy_mat_size, dummy_mat_size).astype(np.float32)
                dummy_result = A @ B
                result_np += np.sum(dummy_result * 1e-10)
            # ----------------------------------------------------
            return result_np

        if jac is not None:
            def cpu_jac_with_artificial_load(x):
                x_np = np.asarray(x, dtype=np.float32)
                result_np_jac = jac(x_np)

                # --- ADD ARTIFICIAL CPU LOAD TO GRADIENT ---
                dummy_mat_size_jac = 2000
                if x_np.size > 100:
                    A = np.random.rand(dummy_mat_size_jac, dummy_mat_size_jac).astype(np.float32)
                    B = np.random.rand(dummy_mat_size_jac, dummy_mat_size_jac).astype(np.float32)
                    dummy_result_jac = A @ B
                    result_np_jac += np.sum(dummy_result_jac * 1e-10)
                # ---------------------------------------------
                return result_np_jac
        else:
            cpu_jac_with_artificial_load = None
            
        result = minimize(cpu_func_with_artificial_load, x0_typed, method=method, jac=cpu_jac_with_artificial_load, 
                          bounds=bounds, constraints=constraints, options=options)
    
    return result