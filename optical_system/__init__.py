"""Fourier-optics simulation framework package.

包内各子模块按需导入（保持 __init__ 轻量，避免 cupy 等重依赖被连带加载）：
- ``optical_system.system``         : OpticalSystem 仿真主类
- ``optical_system.elements``       : 全部光学元件（显式导出）
- ``optical_system.elements_cls``   : 元件基类与核心实现
"""

# 在任何 CuPy 内核编译发生之前，注册 pip 轮子安装的 CUDA DLL 目录
# （Windows + cupy-cuda12x 轮子的必需步骤，见 utils/cuda_path.py）
from utils.cuda_path import ensure_cuda_dll_dirs

ensure_cuda_dll_dirs()
