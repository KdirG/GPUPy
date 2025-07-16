# gpu_support.py
import cupy as cp

#Providing GPU support for the necessary functions

def gradient_gpu(data, dx=1.0):
    """GPU-accelerated gradient computation."""
    data_gpu = cp.asarray(data)
    result_gpu = cp.gradient(data_gpu, dx)
    return cp.asnumpy(result_gpu)
