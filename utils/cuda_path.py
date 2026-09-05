"""确保 pip 轮子安装的 NVIDIA CUDA 库能被 CuPy 在 Windows 上找到。

背景：`cupy-cuda12x` 的 Windows 轮子只会在 ``<CUDA_PATH>/bin`` 与
``cupy/.data/lib`` 中查找 CUDA DLL，不会自动发现 pip 安装的
``site-packages/nvidia/<组件>/bin`` 目录，导致缺 nvrtc/cufft 等 DLL 报错。

即使通过 ``os.add_dll_directory`` 注册，NVRTC 运行时加载其依赖
（nvrtc-builtins）时也不走该注册表，因此这里把所有 ``nvidia/*/bin``
目录前置到 ``PATH``——Loader 的标准 PATH 搜索对依赖解析始终生效。

幂等，可多次调用；非 Windows 平台为 no-op。
"""
import glob
import os
import sys
import sysconfig

_DONE = False


def ensure_cuda_dll_dirs() -> None:
    """把 nvidia-*-cu12 轮子的 bin 目录前置到 PATH（幂等）。"""
    global _DONE
    if _DONE:
        return
    _DONE = True
    if not sys.platform.startswith("win32"):
        return
    purelib = sysconfig.get_paths().get("purelib", "")
    if not purelib:
        return
    bin_dirs = sorted(glob.glob(os.path.join(purelib, "nvidia", "*", "bin")))
    if not bin_dirs:
        return
    # 1) add_dll_directory: 让 ctypes (cupy 的 SoftLink) 能找到 DLL 本体
    #    （Python 3.8+ 的 ctypes 默认不搜索 PATH）
    for dll_dir in bin_dirs:
        try:
            os.add_dll_directory(dll_dir)
        except OSError:
            pass
    # 2) PATH 前置: NVRTC 运行时内部加载依赖 (nvrtc-builtins) 时的兜底搜索路径
    existing_path = os.environ.get("PATH", "")
    parts = [p for p in existing_path.split(os.pathsep) if p and p not in bin_dirs]
    os.environ["PATH"] = os.pathsep.join(bin_dirs + parts)
