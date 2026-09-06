"""pytest 共享 fixture 与 GPU 可用性检测。

GPU 测试统一使用 ``@pytest.mark.gpu`` 标记（在 pyproject.toml 注册）。
GPU 不可用时由 pytest_collection_modifyitems 自动跳过 gpu 标记的用例；
CI 的 gpu job 用 ``pytest -m gpu`` 只选 GPU 用例，CPU job 用 ``-m "not gpu"``。
"""
import sys
from pathlib import Path

# 先注册 pip 轮子的 CUDA DLL 目录（Windows 必需），再导入 cupy
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.cuda_path import ensure_cuda_dll_dirs  # noqa: E402

ensure_cuda_dll_dirs()

import cupy as cp  # noqa: E402
import pytest  # noqa: E402


def _gpu_available() -> bool:
    """检测 CuPy 内核编译链是否真正可用（nvrtc 等就绪）。"""
    try:
        a = cp.array([1.0, -2.0])
        _ = cp.exp(a * 1j)
        return True
    except Exception:
        return False


GPU_OK = _gpu_available()


def pytest_addoption(parser):
    """注册 --golden-update：重新生成黄金图像基准（生成后 skip，便于审查 diff）。"""
    parser.addoption(
        "--golden-update", action="store_true", default=False,
        help="重新生成黄金图像基准哈希（生成后 skip；用 git diff 审查 tests/golden/ 变化）",
    )


def pytest_collection_modifyitems(config, items):
    """GPU 不可用时自动跳过所有 gpu 标记的用例。"""
    if GPU_OK:
        return
    skip = pytest.mark.skip(reason="CuPy GPU 内核链不可用（缺 CUDA/nvrtc）")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
