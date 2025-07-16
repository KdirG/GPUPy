# GPUPy 

**GPUPy** is a high-performance Python library for numerical methods, offering seamless support for both CPU and GPU computations. It is designed for students, researchers, and engineers who need efficient and scalable numerical tools for scientific computing.

## ✨ Features

- ✅ CPU (NumPy) and GPU (CuPy) support with automatic backend switching
- 🔁 Root Finding Methods (e.g. Bisection, Newton-Raphson)
- 🔬 Differentiation Techniques (Forward, Backward, Central Differences)
- ∫ Integration Methods (Trapezoidal, Analytical)
- 📈 Interpolation (Linear and Cubic Spline with GPU versions)
- 🔄 Linear System Solvers (Direct methods, LU Decomposition)
- 🧠 Optimization wrappers (using SciPy)
- 🧮 ODE Solvers (via `scipy.integrate`)
- ⏱ Benchmarking utilities for performance comparison
- 📊 Built-in plotting support for interpolations and function visuals

 🔧 Installation

Clone the repository:



```
git clone https://github.com/yourusername/GPUPy.git
cd GPUPy
Install required dependencies:
pip install -r requirements.txt
```

💡 Make sure you have a compatible CUDA-enabled GPU and cupy installed to use GPU features:
```
pip install cupy
```

📁 Project Structure
```
GPUPy/
├── GPUPy/
│   └── src/
│       └── numerical_methods/
│           ├── core.py
│           ├── root_finding.py
│           ├── differentiation.py
│           ├── integration.py
│           ├── ...
│
├── tests/
│   └── test_root_finding.py
│   └── test_benchmark.py
│
├── examples/
│   └── root_finding_demo.ipynb
│   └── gpu_vs_cpu_benchmark.ipynb
│
├── README.md
└── requirements.txt
```
Usage Example
```
import GPUPy as gp

def f(x): return x**2 - 4
def df(x): return 2 * x

# CPU
root = gp.newton_raphson(f, df, x0=3, use_gpu=False)

# GPU
root_gpu = gp.newton_raphson(f, df, x0=3, use_gpu=True)

print("Root (CPU):", root)
print("Root (GPU):", root_gpu)
```
 Benchmark Example
```
from GPUPy.src.numerical_methods.utils import benchmark
from GPUPy import bisection

def f(x): return x**2 - 4

cpu_time = benchmark(bisection, f, a=0, b=5, use_gpu=False)
gpu_time = benchmark(bisection, f, a=0, b=5, use_gpu=True)

print(f"CPU: {cpu_time:.6f}s | GPU: {gpu_time:.6f}s")
```
🤝 Contributing
Pull requests are welcome! If you want to contribute:

Fork the repo

Create a new branch

Add your feature or fix

Submit a PR 🚀

📜 License
MIT License. See LICENSE for more details.
