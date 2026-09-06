# 快速上手

## 安装

```bash
# 1. 创建虚拟环境（Python 3.10+）
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux

# 2. 安装依赖（含 CUDA 12 运行库 pip 轮子，无需单独装 CUDA Toolkit）
pip install -r requirements.txt

# 3. editable 安装框架
pip install -e .
```

GPU 要求：NVIDIA 显卡 + 驱动。Windows 上 cupy 轮子不会自动发现 pip 安装的
CUDA 库，本仓库通过 `utils/cuda_path.py`（导入 `optical_system` 时自动执行）
解决，无需手动设置。

## 第一个仿真

见[首页](index.md)。受支持的示例脚本（物理结论随版本验证）：

```bash
python examples/radial_polarization_focusing.py    # 矢量：径向偏振高NA聚焦
python examples/qplate_vector_vortex.py            # 矢量：q-plate 矢量涡旋
python examples/broadband_chromatic_shift.py       # 工作流：宽谱色差焦移
python examples/multilayer_slab_superlens.py       # 多层膜：超透镜传函
```

历史探索脚本在 `examples/legacy/`（不保证随当前版本运行），完整索引见
`examples/README.md`。

## 矢量引擎入门

把标量场换成偏振受控的矢量场，其余编排不变：

```python
from vector.sources import vector_gaussian
from vector.elements import HalfWavePlate
from vector.system import VectorOpticalSystem

src = vector_gaussian(x, x, wavelength, waist, 'LCP')    # 圆偏振高斯
vsys = VectorOpticalSystem(wavelength, x, x, src)
vsys.add_element(HalfWavePlate(10.0, fast_axis_angle=0.7))
planes = vsys.propagate_to_cross_sections([0.0, 25.0])   # 各平面含 (Ex,Ey,Ez)
```

## 关键约定

!!! warning "调制函数必须用 cupy ufunc"
    `SpatialPlate` / `MomentumSpacePlate` 的 `modulation_function` 接收
    **cupy 数组**，内部须用 `cp.exp` / `cp.arctan2` 等，numpy ufunc 无法
    分派到 GPU（会直接 TypeError）。

!!! note "数值精度"
    `OpticalSystem` 默认 `complex128`/`float64`（研究精度）。
    大网格（如 10241×10241）在 8GB 显存下会 OOM，显式传
    `dtype=cp.complex64`。

!!! note "场归一化"
    `OpticalSystem.__init__` 把 `sum|U|²` 归一，所有强度是相对单位。

## 常见陷阱

| 现象 | 原因 | 处理 |
|------|------|------|
| `CuPy failed to load nvrtc...` | CUDA 库未被发现 | `from utils.cuda_path import ensure_cuda_dll_dirs; ensure_cuda_dll_dirs()` |
| 调制函数 TypeError | numpy ufunc on cupy 数组 | 换 `cp.*` |
| 长距离传播出现周期性 wrap-around | 仿真窗口小于光束路径 | 加大窗口或分步传播 |
| 动量空间角度刻度异常 | `Plotter.wavelength` 与仿真不一致 | 构造 Plotter 时显式传 `wavelength` |

## 测试

```bash
python -m pytest tests/ -m "not gpu"   # CPU 部分
python -m pytest tests/                # 全量（GPU 不可用时自动跳过 gpu 标记）
python -m pytest tests/ -m gpu         # 仅 GPU 物理回归
```
