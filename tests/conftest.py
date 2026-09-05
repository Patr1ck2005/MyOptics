"""pytest 共享 fixture 与 GPU 可用性检测。"""
import cupy as cp
import pytest


def _gpu_available() -> bool:
    """检测 CuPy 内核编译链是否真正可用（nvrtc 等就绪）。"""
    try:
        a = cp.array([1.0, -2.0])
        _ = cp.exp(a * 1j)
        return True
    except Exception:
        return False


GPU_OK = _gpu_available()

requires_gpu = pytest.mark.skipif(not GPU_OK, reason="CuPy GPU 内核链不可用（缺 CUDA/nvrtc）")
