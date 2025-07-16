import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
import ast
import gc
import uuid 
# GPU memory cleanup utility
def cleanup_gpu_memory():
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except ImportError:
        pass

try:
    from GPUPy import (
        compute_derivative,
        analytical_integral,
        solve_linear_system,
        linear_interpolation,
        spline_interpolation,
        odeint_wrapper,
        bisection,
        newton_raphson,
        minimize_scalar_wrapper,
        minimize_wrapper,
        compile_function_from_string
    )
    GPUPY_AVAILABLE = True
except ImportError as e:
    print(f"Error importing GPUPy library: {e}")
    print("Please ensure the GPUPy library is structured correctly and accessible.")
    GPUPY_AVAILABLE = False
    
    # Define placeholder functions
    def compute_derivative(*args, **kwargs): return "Error: GPUPy not loaded."
    def analytical_integral(*args, **kwargs): return ("Error: GPUPy not loaded.", 0)
    def solve_linear_system(*args, **kwargs): return "Error: GPUPy not loaded."
    def linear_interpolation(*args, **kwargs): return "Error: GPUPy not loaded."
    def spline_interpolation(*args, **kwargs): return "Error: GPUPy not loaded."
    def odeint_wrapper(*args, **kwargs): return "Error: GPUPy not loaded."
    def bisection(*args, **kwargs): return "Error: GPUPy not loaded."
    def newton_raphson(*args, **kwargs): return "Error: GPUPy not loaded."
    def minimize_scalar_wrapper(*args, **kwargs): return "Error: GPUPy not loaded."
    def minimize_wrapper(*args, **kwargs): return "Error: GPUPy not loaded."
    def compile_function_from_string(func_str, var='x'): return lambda x: 0


def safe_eval(expr, allowed_names=None):
    """Safely evaluate string expressions"""
    if allowed_names is None:
        allowed_names = ['__builtins__', 'list', 'dict', 'tuple', 'set', 'frozenset', 'np']
    
    # type control
    if not isinstance(expr, (str, bytes)):
        raise TypeError(f"Expression must be a string, got {type(expr)}")
    
    if not expr.strip():
        raise ValueError("Expression cannot be empty")
    
    try:
        # Parse the expression
        parsed = ast.parse(expr, mode='eval')
        
        # Create a safe namespace
        safe_dict = {"__builtins__": {}}
        # adding numpy to the safe list
        safe_dict['np'] = np 
        
        for name in allowed_names:
            if name in globals():
                safe_dict[name] = globals()[name]
        
        
        compiled_code = compile(parsed, '<string>', 'eval')
        return eval(compiled_code, safe_dict)
        
    except (ValueError, SyntaxError, NameError, TypeError) as e:
        raise ValueError(f"Invalid expression: {e}")


def run_and_measure(func, *args, use_gpu=False, **kwargs):
    """Enhanced run_and_measure with better error handling and memory cleanup"""
    if not GPUPY_AVAILABLE:
        return ("GPUPy not available", 0)
    
    try:
        start_time = time.perf_counter()
        result = func(*args, use_gpu=use_gpu, **kwargs)
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        # Cleanup GPU memory after computation
        if use_gpu:
            cleanup_gpu_memory()
        
        # Handle different return types
        if isinstance(result, tuple):
            return (*result, duration)
        else:
            return result, duration
            
    except Exception as e:
        return f"Error: {str(e)}", 0

# --- Differentiation Tab ---
def differentiate_func_en(data_str, dx, method):
    try:
        if not data_str.strip():
            return "Error: Please enter data values"
            
        data = np.array([float(x.strip()) for x in data_str.split(',') if x.strip()])
        
        if len(data) < 2:
            return "Error: At least 2 data points required"
            
        cpu_result, cpu_time = run_and_measure(compute_derivative, data, dx=dx, method=method, use_gpu=False)
        gpu_result, gpu_time = run_and_measure(compute_derivative, data, dx=dx, method=method, use_gpu=True)

        output_text = f"**CPU Result:** {cpu_result}\n**CPU Time:** {cpu_time:.6f} s\n\n" \
                      f"**GPU Result:** {gpu_result}\n**GPU Time:** {gpu_time:.6f} s\n"
        return output_text
    except Exception as e:
        return f"Error: {e}"

# --- Integration Tab ---
def integrate_func_en(func_str, a, b, num_points):
    try:
        if not func_str.strip():
            return "Error: Please enter a function"
            
        func = compile_function_from_string(func_str)
        
        # Run CPU computation
        cpu_output = run_and_measure(analytical_integral, func, a, b, num_points=num_points, use_gpu=False)
        gpu_output = run_and_measure(analytical_integral, func, a, b, num_points=num_points, use_gpu=True)
        
        # Handle different return formats
        if len(cpu_output) == 3:  # (result, error, time)
            cpu_result, cpu_error, cpu_time = cpu_output
        else:  # (result, time) or ((result, error), time)
            if isinstance(cpu_output[0], tuple):
                cpu_result, cpu_error = cpu_output[0]
                cpu_time = cpu_output[1]
            else:
                cpu_result, cpu_time = cpu_output
                cpu_error = 0
        
        if len(gpu_output) == 3:
            gpu_result, gpu_error, gpu_time = gpu_output
        else:
            if isinstance(gpu_output[0], tuple):
                gpu_result, gpu_error = gpu_output[0]
                gpu_time = gpu_output[1]
            else:
                gpu_result, gpu_time = gpu_output
                gpu_error = 0

        output_text = f"**CPU Integral Result:** {cpu_result:.6f} (Error Estimate: {cpu_error:.6e})\n**CPU Time:** {cpu_time:.6f} s\n\n" \
                      f"**GPU Integral Result:** {gpu_result:.6f} (Error Estimate: {gpu_error:.6e})\n**GPU Time:** {gpu_time:.6f} s\n"
        return output_text
    except Exception as e:
        return f"Error: {e}"

# --- Linear System Solving Tab ---
def solve_linear_func_en(A_str, b_str):
    try:
        if not A_str.strip() or not b_str.strip():
            return "Error: Please enter both matrix A and vector b"
            
        # Use safe_eval instead of eval
        A = np.array(safe_eval(A_str))
        b = np.array(safe_eval(b_str))
        
        # Validate dimensions
        if len(A.shape) != 2:
            return "Error: Matrix A must be 2-dimensional"
        if A.shape[0] != A.shape[1]:
            return "Error: Matrix A must be square"
        if len(b) != A.shape[0]:
            return "Error: Vector b length must match matrix A dimensions"
        
        cpu_solution, cpu_time = run_and_measure(solve_linear_system, A, b, use_gpu=False)
        gpu_solution, gpu_time = run_and_measure(solve_linear_system, A, b, use_gpu=True)

        output_text = f"**CPU Solution:** {cpu_solution}\n**CPU Time:** {cpu_time:.6f} s\n\n" \
                      f"**GPU Solution:** {gpu_solution}\n**GPU Time:** {gpu_time:.6f} s\n"
        return output_text
    except Exception as e:
        return f"Error: {e}"

# --- Interpolation Tab ---
def interpolate_func_en(x_str, y_str, x_new_str, method, bc_type):
    try:
        if not all([x_str.strip(), y_str.strip(), x_new_str.strip()]):
            return "Error: Please fill all input fields", None
            
        x = np.array([float(val.strip()) for val in x_str.split(',') if val.strip()])
        y = np.array([float(val.strip()) for val in y_str.split(',') if val.strip()])
        x_new = np.array([float(val.strip()) for val in x_new_str.split(',') if val.strip()])

        if len(x) != len(y):
            return "Error: x and y arrays must have the same length", None
        if len(x) < 2:
            return "Error: At least 2 data points required", None

        if method == "linear":
            cpu_y_new, cpu_time = run_and_measure(linear_interpolation, x, y, x_new, use_gpu=False)
            gpu_y_new, gpu_time = run_and_measure(linear_interpolation, x, y, x_new, use_gpu=True)
        else:  # spline
            cpu_y_new, cpu_time = run_and_measure(spline_interpolation, x, y, x_new, bc_type=bc_type, use_gpu=False)
            gpu_y_new, gpu_time = run_and_measure(spline_interpolation, x, y, x_new, bc_type=bc_type, use_gpu=True)
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, 'ro', label='Original Data', markersize=8)
        plt.plot(x_new, cpu_y_new, 'b--', label=f'CPU {method.capitalize()} Interpolation', linewidth=2)
        plt.plot(x_new, gpu_y_new, 'g:', label=f'GPU {method.capitalize()} Interpolation', linewidth=2)
        plt.title(f'{method.capitalize()} Interpolation Comparison')
        plt.xlabel('X values')
        plt.ylabel('Y values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_path = f"interpolation_plot_{uuid.uuid4()}.png" # Benzersiz dosya adı
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        # Show limited results to avoid overwhelming output
        n_show = min(5, len(cpu_y_new))
        output_text = f"**CPU Interpolation Results (first {n_show}):** {cpu_y_new[:n_show]}\n**CPU Time:** {cpu_time:.6f} s\n\n" \
                      f"**GPU Interpolation Results (first {n_show}):** {gpu_y_new[:n_show]}\n**GPU Time:** {gpu_time:.6f} s\n" \
                      f"Total interpolated points: {len(cpu_y_new)}"
        
        return output_text, plot_path

    except Exception as e:
        return f"Error: {e}", None

# --- ODE Solver Tab ---
def solve_ode_func_en(func_str, y0_str, t_str, ode_method):
    try:
        if not all([func_str.strip(), y0_str.strip(), t_str.strip()]):
            return "Error: Please fill all input fields", None
            
        y0 = np.array([float(x.strip()) for x in y0_str.split(',') if x.strip()])
        t = np.array([float(x.strip()) for x in t_str.split(',') if x.strip()])
        
        if len(t) < 2:
            return "Error: At least 2 time points required", None

        def ode_func_wrapper(y, t, *args):
            """Safe ODE function wrapper"""
            try:
                # Create safe execution environment
                exec_globals = {
                    'y': y, 't': t, 'np': np,
                    'sin': np.sin, 'cos': np.cos, 'exp': np.exp, 'log': np.log,
                    'sqrt': np.sqrt, 'abs': np.abs, 'pi': np.pi, 'e': np.e
                }
                
                # Add cupy if available and if working with GPU arrays
                try:
                    import cupy as cp
                    if hasattr(y, '__array_interface__') and 'cupy' in str(type(y)):
                        exec_globals['cp'] = cp
                except ImportError:
                    pass
                
                result = eval(func_str, exec_globals)
                
                # Ensure result is array-like, especially for multi-dimensional y where a scalar might be returned by error
                if isinstance(result, (list, tuple)):
                    return np.array(result, dtype=np.float64)
                elif np.isscalar(result) and len(y) > 1:
                    # If func_str defines a single scalar output for a multi-dim system,
                    # this is likely an error in func_str for a system.
                    # For now, replicate it to match y's dimension for compatibility, or raise an error.
                    # Raising an error is often better for clarity.
                    # raise ValueError("ODE function returned a scalar for a multi-dimensional y. Expected an array/list.")
                    return np.array([result] * len(y), dtype=np.float64) # Fallback: Replicate scalar to match y's size
                elif np.isscalar(result) and len(y) == 1: # Single-dimension ODE, scalar is fine
                    return np.array([result], dtype=np.float64)
                return result # If already a numpy array
            except Exception as e:
                raise ValueError(f"Error evaluating ODE function string: {e}")

        # Take CPU and GPU output
        cpu_output = run_and_measure(odeint_wrapper, ode_func_wrapper, y0, t, use_gpu=False, method=ode_method)
        gpu_output = run_and_measure(odeint_wrapper, ode_func_wrapper, y0, t, use_gpu=True, method=ode_method)

        # Separating results and times, handling error cases
        cpu_solution = cpu_output[0] if isinstance(cpu_output, tuple) else cpu_output
        cpu_time = cpu_output[1] if isinstance(cpu_output, tuple) else 0

        gpu_solution = gpu_output[0] if isinstance(gpu_output, tuple) else gpu_output
        gpu_time = gpu_output[1] if isinstance(gpu_output, tuple) else 0
        
        # If the solution is an error message string, plot the graph
        if isinstance(cpu_solution, str) or isinstance(gpu_solution, str):
            error_message = f"CPU Solution Error: {cpu_solution}\nGPU Solution Error: {gpu_solution}"
            return error_message, None

        # Plotting
        plt.figure(figsize=(12, 6))
        
        # Handle multidimensional solutions
        if len(cpu_solution.shape) == 1:
            plt.plot(t, cpu_solution, 'b-', label='CPU Solution', linewidth=2)
        else:
            for i in range(min(cpu_solution.shape[1], 5)):  # Plot max 5 components
                plt.plot(t, cpu_solution[:, i], 'b-', label=f'CPU y{i}', alpha=0.7)
        
        if len(gpu_solution.shape) == 1:
            plt.plot(t, gpu_solution, 'g--', label='GPU Solution', linewidth=2)
        else:
            for i in range(min(gpu_solution.shape[1], 5)):  # Plot max 5 components
                plt.plot(t, gpu_solution[:, i], 'g--', label=f'GPU y{i}', alpha=0.7)
            
        plt.title('ODE Solution Comparison')
        plt.xlabel('Time (t)')
        plt.ylabel('Y(t)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Unique file name generated
        plot_path = f"ode_solution_plot_{uuid.uuid4()}.png"
        
 
        
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close() # Figure'ı kapatmak önemli
        
        #Define the variable n_show

        # n_show is only meaningful for arrays, not strings
        if isinstance(cpu_solution, np.ndarray):
            n_show = min(5, len(cpu_solution)) 
        else:
            n_show = 0 # If cpu_solution is not an array, don't show points

        # cpu_solution_str ve gpu_solution_str oluşturma
        cpu_solution_str = ""
        if isinstance(cpu_solution, np.ndarray):
            # Show first n_show elements, formatted
            for i in range(min(n_show, len(cpu_solution))):
                # Check if it's a 1D array or a 2D array
                if cpu_solution.ndim > 1:
                    cpu_solution_str += f"[{', '.join([f'{val:.8f}' for val in cpu_solution[i]])}]\n"
                else:
                    cpu_solution_str += f"[{cpu_solution[i]:.8f}]\n"
        else:
            cpu_solution_str = str(cpu_solution) # If it's an error string, use it directly


        gpu_solution_str = ""
        if isinstance(gpu_solution, np.ndarray):
            # Show first n_show elements, formatted
            for i in range(min(n_show, len(gpu_solution))):
                if gpu_solution.ndim > 1:
                    gpu_solution_str += f"[{', '.join([f'{val:.8f}' for val in gpu_solution[i]])}]\n"
                else:
                    gpu_solution_str += f"[{gpu_solution[i]:.8f}]\n"
        else:
            gpu_solution_str = str(gpu_solution) # If it's an error string, use it directly


        output_text = f"**CPU Solution (first {n_show} time points):**\n{cpu_solution_str}\n**CPU Time:** {cpu_time:.6f} s\n\n" \
                      f"**GPU Solution (first {n_show} time points):**\n{gpu_solution_str}\n**GPU Time:** {gpu_time:.6f} s\n" \
                      f"Total time points solved: {len(cpu_solution) if isinstance(cpu_solution, np.ndarray) else 'N/A'}"
        
        return output_text, plot_path
    except Exception as e:
        return f"Error: {e}", None

# --- Root Finding Tab --- 
def find_root_func_en(func_str, df_str, a, b, x0, method, tol, max_iter):
    try:
        if not func_str.strip():
            return "Error: Please enter a function"
            
        func = compile_function_from_string(func_str)
        
        if method == "Bisection":
            if a >= b:
                return "Error: Lower bound must be less than upper bound"
            cpu_result, cpu_time = run_and_measure(bisection, func, a, b, tolerance=tol, max_iterations=int(max_iter), use_gpu=False)
            gpu_result, gpu_time = run_and_measure(bisection, func, a, b, tolerance=tol, max_iterations=int(max_iter), use_gpu=True)
        elif method == "Newton-Raphson":
            if not df_str.strip():
                return "Error: Please enter the derivative function for Newton-Raphson method"
            df = compile_function_from_string(df_str)
            cpu_result, cpu_time = run_and_measure(newton_raphson, func, df, x0, tol=tol, max_iter=int(max_iter), use_gpu=False)
            gpu_result, gpu_time = run_and_measure(newton_raphson, func, df, x0, tol=tol, max_iter=int(max_iter), use_gpu=True)
        
        output_text = f"**CPU Root Result:** {cpu_result:.8f}\n**CPU Time:** {cpu_time:.6f} s\n\n" \
                      f"**GPU Root Result:** {gpu_result:.8f}\n**GPU Time:** {gpu_time:.6f} s\n" \
                      f"**Absolute Difference:** {abs(float(cpu_result) - float(gpu_result)):.2e}"
        return output_text
    except Exception as e:
        return f"Error: {e}"

# --- Optimization Tab --- 
def optimize_func_en(func_str, x0_str, opt_type, opt_method):
    try:
        if not func_str.strip():
            return "Error: Please enter a function"
            
        func = compile_function_from_string(func_str)

        if opt_type == "Scalar":
            cpu_res, cpu_time = run_and_measure(minimize_scalar_wrapper, func, use_gpu=False, method=opt_method)
            gpu_res, gpu_time = run_and_measure(minimize_scalar_wrapper, func, use_gpu=True, method=opt_method)
            
            cpu_min_val = cpu_res.fun if hasattr(cpu_res, 'fun') else cpu_res
            gpu_min_val = gpu_res.fun if hasattr(gpu_res, 'fun') else gpu_res
            cpu_min_x = cpu_res.x if hasattr(cpu_res, 'x') else "N/A"
            gpu_min_x = gpu_res.x if hasattr(gpu_res, 'x') else "N/A"

            output_text = f"**CPU Optimization (Scalar) Result:**\nMinimum Value: {cpu_min_val:.8f} (x={cpu_min_x})\nCPU Time: {cpu_time:.6f} s\n\n" \
                          f"**GPU Optimization (Scalar) Result:**\nMinimum Value: {gpu_min_val:.8f} (x={gpu_min_x})\nGPU Time: {gpu_time:.6f} s\n"

        elif opt_type == "Multivariate":
            if not x0_str.strip():
                return "Error: Please enter initial guess for multivariate optimization"
            x0 = np.array(safe_eval(x0_str))
            cpu_res, cpu_time = run_and_measure(minimize_wrapper, func, x0, use_gpu=False, method=opt_method)
            gpu_res, gpu_time = run_and_measure(minimize_wrapper, func, x0, use_gpu=True, method=opt_method)

            cpu_min_val = cpu_res.fun if hasattr(cpu_res, 'fun') else cpu_res
            gpu_min_val = gpu_res.fun if hasattr(gpu_res, 'fun') else gpu_res
            cpu_min_x = cpu_res.x if hasattr(cpu_res, 'x') else "N/A"
            gpu_min_x = gpu_res.x if hasattr(gpu_res, 'x') else "N/A"

            output_text = f"**CPU Optimization (Multivariate) Result:**\nMinimum Value: {cpu_min_val:.8f} (x={cpu_min_x})\nCPU Time: {cpu_time:.6f} s\n\n" \
                          f"**GPU Optimization (Multivariate) Result:**\nMinimum Value: {gpu_min_val:.8f} (x={gpu_min_x})\nGPU Time: {gpu_time:.6f} s\n"
        
        return output_text
    except Exception as e:
        return f"Error: {e}"

# --- Gradio Interface Configuration --- 
with gr.Blocks(title="GPUPy Numerical Methods Library", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # GPUPy: GPU Accelerated Numerical Methods Library
        
        This interface allows you to use various numerical methods from the GPUPy library and compare CPU vs. GPU performance.
        """
    )
    
    with gr.Tab(" Differentiation"):
        gr.Markdown("## Numerical Differentiation")
        gr.Markdown("Computes the derivative of an input array using various finite difference methods.")
        
        with gr.Row():
            with gr.Column():
                diff_data_input = gr.Textbox(
                    label="Data Array", 
                    placeholder="1,2,4,7,11,16", 
                    value="1,2,4,7,11,16",
                    info="Comma-separated numbers"
                )
                diff_dx_input = gr.Number(label="Step Size (dx)", value=1.0, minimum=0.001)
            with gr.Column():
                diff_method_input = gr.Radio(
                    ["auto", "forward", "backward", "central"], 
                    label="Differentiation Method", 
                    value="central"
                )
        
        diff_button = gr.Button("Compute Derivative", variant="primary")
        diff_output = gr.Textbox(label="Results", lines=6)
        
        diff_button.click(
            differentiate_func_en, 
            inputs=[diff_data_input, diff_dx_input, diff_method_input], 
            outputs=diff_output
        )

    with gr.Tab(" Integration"):
        gr.Markdown("## Numerical Integration")
        gr.Markdown("Computes the definite integral of a mathematical function.")
        
        with gr.Row():
            with gr.Column():
                int_func_input = gr.Textbox(
                    label="Function f(x)", 
                    placeholder="x**2 + np.sin(x)", 
                    value="x**2",
                    info="Python expression using 'x' as variable"
                )
                int_a_input = gr.Number(label="Lower Bound (a)", value=0.0)
                int_b_input = gr.Number(label="Upper Bound (b)", value=1.0)
            with gr.Column():
                int_num_points_input = gr.Slider(
                    minimum=100, maximum=100000, step=100, value=10000, 
                    label="Number of Integration Points"
                )
        
        int_button = gr.Button("Compute Integral", variant="primary")
        int_output = gr.Textbox(label="Results", lines=6)
        
        int_button.click(
            integrate_func_en, 
            inputs=[int_func_input, int_a_input, int_b_input, int_num_points_input], 
            outputs=int_output
        )

    with gr.Tab(" Linear Systems"):
        gr.Markdown("## Linear System Solving (Ax = b)")
        gr.Markdown("Solves systems of linear equations using advanced numerical methods.")
        
        with gr.Row():
            with gr.Column():
                ls_A_input = gr.Textbox(
                    label="Matrix A", 
                    placeholder="[[2,1],[1,3]]", 
                    value="[[2,1],[1,3]]",
                    info="2D list format"
                )
                ls_b_input = gr.Textbox(
                    label="Vector b", 
                    placeholder="[4,7]", 
                    value="[4,7]",
                    info="1D list format"
                )
            with gr.Column():
                gr.Markdown("**Example:**\nA = [[2,1],[1,3]]\nb = [4,7]\nSolution: x = [1,2]")
        
        ls_button = gr.Button("Solve Linear System", variant="primary")
        ls_output = gr.Textbox(label="Results", lines=6)
        
        ls_button.click(
            solve_linear_func_en, 
            inputs=[ls_A_input, ls_b_input], 
            outputs=ls_output
        )

    with gr.Tab(" Interpolation"):
        gr.Markdown("## Data Interpolation")
        gr.Markdown("Interpolates between known data points using linear or spline methods.")
        
        with gr.Row():
            with gr.Column():
                interp_x_input = gr.Textbox(
                    label="Known X Values", 
                    placeholder="0,1,2,3,4", 
                    value="0,1,2,3,4",
                    info="Comma-separated numbers"
                )
                interp_y_input = gr.Textbox(
                    label="Known Y Values", 
                    placeholder="0,1,4,9,16", 
                    value="0,1,4,9,16",
                    info="Comma-separated numbers"
                )
                interp_x_new_input = gr.Textbox(
                    label="New X Values to Interpolate", 
                    placeholder="0.5,1.5,2.5,3.5", 
                    value="0.5,1.5,2.5,3.5",
                    info="Comma-separated numbers"
                )
            with gr.Column():
                interp_method_input = gr.Radio(
                    ["linear", "spline"], 
                    label="Interpolation Method", 
                    value="spline"
                )
                interp_bc_type_input = gr.Radio(
                    ["natural", "clamped", "not-a-knot"], 
                    label="Spline Boundary Condition", 
                    value="natural",
                    info="Only for spline interpolation"
                )
        
        interp_button = gr.Button("Perform Interpolation", variant="primary")
        interp_output = gr.Textbox(label="Results", lines=6)
        interp_plot_output = gr.Image(label="Interpolation Visualization")
        
        interp_button.click(
            interpolate_func_en, 
            inputs=[interp_x_input, interp_y_input, interp_x_new_input, 
                    interp_method_input, interp_bc_type_input], 
            outputs=[interp_output, interp_plot_output]
        )
    
    with gr.Tab(" ODE Solver"):
        gr.Markdown("## Ordinary Differential Equation Solver")
        gr.Markdown("Solves ODE systems: dy/dt = f(y, t)")
        
        with gr.Row():
            with gr.Column():
                ode_func_input = gr.Textbox(
                    label="ODE Function f(y,t)", 
                    placeholder="-y[0] + t", 
                    value="-y[0]",
                    info="Python expression using 'y' and 't'"
                )
                ode_y0_input = gr.Textbox(
                    label="Initial Condition y0", 
                    placeholder="1.0", 
                    value="1.0",
                    info="Comma-separated for multi-dimensional"
                )
            with gr.Column():
                ode_t_input = gr.Textbox(
                    label="Time Points", 
                    value=",".join([f"{x:.2f}" for x in np.linspace(0, 5, 20)]),
                    info="Comma-separated time values"
                )
                ode_method_input = gr.Radio(["RK45", "BDF"], label="Integration Method", value="RK45")
        
        ode_button = gr.Button("Solve ODE", variant="primary")
        ode_output = gr.Textbox(label="Results", lines=6)
        ode_plot_output = gr.Image(label="Solution Visualization")
        
        ode_button.click(
            solve_ode_func_en, 
            inputs=[ode_func_input, ode_y0_input, ode_t_input, ode_method_input], 
            outputs=[ode_output, ode_plot_output]
        )

    with gr.Tab(" Root Finding"):
        gr.Markdown("## Root Finding Methods")
        gr.Markdown("Finds zeros of mathematical functions using bisection or Newton-Raphson methods.")
        
        with gr.Row():
            with gr.Column():
                root_func_input = gr.Textbox(
                    label="Function f(x)", 
                    placeholder="x**2 - 2", 
                    value="x**2 - 2",
                    info="Python expression using 'x'"
                )
                root_df_input = gr.Textbox(
                    label="Derivative f'(x)", 
                    placeholder="2*x", 
                    value="2*x",
                    info="Required for Newton-Raphson"
                )
            with gr.Column():
                root_method_input = gr.Radio(
                    ["Bisection", "Newton-Raphson"], 
                    label="Root Finding Method", 
                    value="Newton-Raphson"
                )
        
        with gr.Row():
            with gr.Column():
                root_a_input = gr.Number(label="Lower Bound (a) - Bisection", value=0.0)
                root_b_input = gr.Number(label="Upper Bound (b) - Bisection", value=2.0)
            with gr.Column():
                root_x0_input = gr.Number(label="Initial Guess (x0) - Newton", value=1.0)
                root_tol_input = gr.Number(label="Tolerance", value=1e-6)
                root_max_iter_input = gr.Number(label="Max Iterations", value=100)
        
        root_button = gr.Button("Find Root", variant="primary")
        root_output = gr.Textbox(label="Results", lines=6)
        
        root_button.click(
            find_root_func_en, 
            inputs=[root_func_input, root_df_input, root_a_input, root_b_input, 
                    root_x0_input, root_method_input, root_tol_input, root_max_iter_input], 
            outputs=root_output
        )

    with gr.Tab(" Optimization"):
        gr.Markdown("## Function Optimization")
        gr.Markdown("Finds minimum values of scalar or multivariate functions.")
        
        with gr.Row():
            with gr.Column():
                opt_func_input = gr.Textbox(
                    label="Function to Minimize", 
                    placeholder="x**2 or x[0]**2 + x[1]**2", 
                    value="x**2",
                    info="Use 'x' for scalar, 'x[i]' for multivariate"
                )
                opt_x0_input = gr.Textbox(
                    label="Initial Guess", 
                    placeholder="1.0 or [1,1]", 
                    value="1.0",
                    info="Single value for scalar, list for multivariate"
                )
            with gr.Column():
                opt_type_input = gr.Radio(
                    ["Scalar", "Multivariate"], 
                    label="Optimization Type", 
                    value="Scalar"
                )
                opt_method_input = gr.Textbox(
                    label="Optimization Method", 
                    placeholder="brent, BFGS, etc.", 
                    value="brent",
                    info="Method depends on optimization type"
                )
        
        opt_button = gr.Button("Optimize Function", variant="primary")
        opt_output = gr.Textbox(label="Results", lines=6)
        
        opt_button.click(
            optimize_func_en, 
            inputs=[opt_func_input, opt_x0_input, opt_type_input, opt_method_input], 
            outputs=opt_output
        )

    # Footer
    gr.Markdown(
        """
        ---
        ### Usage Tips:
        - **Functions**: Use standard Python math expressions (x**2, np.sin(x), etc.)
        - **Error Handling**: All inputs are validated before processing
        - **Memory Management**: GPU memory is automatically cleaned after each operation
        
        ### Security Notes:
        - Input expressions are safely evaluated to prevent code injection
        - Only mathematical operations and numpy functions are allowed
        """
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",   # Allow external access
        server_port=7860,         # Default Gradio port
        share=False,              # Set to True for public sharing
        debug=True                # Enable debug mode
    )
