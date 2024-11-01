import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.fft import fft2 as scipy_fft2

# 检查是否可以使用 pyFFTW 和 CuPy
try:
    import pyfftw
    pyfftw_available = True
except ImportError:
    pyfftw_available = False

try:
    import cupy as cp
    cupy_available = True
except ImportError:
    cupy_available = False

# 参数设置
mesh = 1024*10
size = 1024*2  # 仿真的尺寸
w0 = 10  # 高斯光束的宽度
vortex_charge = 1  # 相位涡旋的阶数

# 生成相位涡旋的高斯光束
def generate_vortex_gaussian(size, mesh, w0, charge):
    x = np.linspace(-size // 2, size // 2, mesh)
    y = np.linspace(-size // 2, size // 2, mesh)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    gaussian_beam = np.exp(-(R / w0) ** 2) * np.exp(1j * charge * theta)
    return gaussian_beam

# 生成测试数据
data = generate_vortex_gaussian(size, mesh, w0, vortex_charge)

# 计时函数
def benchmark_fft(library_name, fft_func, data):
    start_time = time.time()
    result = fft_func(data)
    elapsed_time = time.time() - start_time
    print(f"{library_name} FFT time: {elapsed_time:.6f} seconds")
    return result, elapsed_time

# NumPy FFT
fft_numpy, time_numpy = benchmark_fft("NumPy", np.fft.fft2, data)

# SciPy FFT
fft_scipy, time_scipy = benchmark_fft("SciPy", scipy_fft2, data)

# pyFFTW FFT
if pyfftw_available:
    data_fftw = pyfftw.empty_aligned((size, size), dtype='complex128')
    data_fftw[:] = data
    fftw_fft2 = pyfftw.builders.fft2(data_fftw)
    fft_fftw, time_fftw = benchmark_fft("pyFFTW", fftw_fft2, data_fftw)
else:
    fft_fftw = None
    time_fftw = None

# CuPy FFT
if cupy_available:
    data_cupy = cp.array(data)
    fft_cupy, time_cupy = benchmark_fft("CuPy", cp.fft.fft2, data_cupy)
    fft_cupy = cp.asnumpy(fft_cupy)  # 将结果转换回 CPU 以便显示
else:
    fft_cupy = None
    time_cupy = None

# 可视化
def plot_fft_result(fft_result, title):
    plt.figure(figsize=(6, 6))
    # 使用 fftshift 将零频移到中心，取幅度谱进行显示
    plt.imshow(np.abs(np.fft.fftshift(fft_result)), cmap='viridis', extent=[-size//2, size//2, -size//2, size//2])
    plt.colorbar()
    plt.title(title)
    plt.xlabel('kx')
    plt.ylabel('ky')

# 绘制结果
plot_fft_result(fft_numpy, f"NumPy FFT (Time: {time_numpy:.6f}s)")
plot_fft_result(fft_scipy, f"SciPy FFT (Time: {time_scipy:.6f}s)")

if fft_fftw is not None:
    plot_fft_result(fft_fftw, f"pyFFTW FFT (Time: {time_fftw:.6f}s)")

if fft_cupy is not None:
    plot_fft_result(fft_cupy, f"CuPy FFT (Time: {time_cupy:.6f}s)")

plt.show()
