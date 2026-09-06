# MyOptics

MyOptics 是一个 GPU 加速的傅里叶光学仿真框架，核心是角谱法（Angular Spectrum
Method）传播器与可组合的光学元件系统：多层膜 TMM 传递函数作为动量空间
元件嵌入光路（超透镜传函仿真）；全矢量引擎处理偏振与纵向场；工作流层
支持参数扫描与宽谱合成。

## 核心能力

- **角谱传播**：严格（含倏逝波）/ Fresnel 双模式，缓存传播器
  （`AngularSpectrumPropagator`），纵向多步扫描提速 ~2.7×
- **元件库**：透镜（近轴/非近轴高 NA）、光阑（6 种）、光栅（4 种）、
  轴棱锥、动量空间调制器（MSPP）
- **多层膜耦合**：`MultilayerSlab` 把 TMM 场传函 H(kx,ky) 作用于光场，
  传播带透过率与倏逝增强（超透镜机制）一次到位
- **矢量引擎**（`vector/`）：三分量场 (Ex,Ey,Ez) 严格横场投影传播、
  Richards-Wolf 高 NA 矢量焦场、Jones 偏振元件（波片/q-plate 自旋-轨道
  耦合）、`VectorOpticalSystem` 端到端矢量光路
- **工作流层**（`workflows/`）：`Field` 结果容器（npz 持久化）、
  `ParameterSweep` 参数扫描 + 像质指标（FWHM/质心）、`SimSpectrum`
  宽谱合成（非相干/相干）
- **可视化**：横截面 / 纵截面 / 动量空间，LogNorm 阈值显示

## 安装

```bash
pip install -r requirements.txt   # 或 pip install -e .[dev]
pip install -e .
```

详见 [快速上手](quickstart.md)。

## 最小示例

```python
import cupy as cp
import numpy as np
from optical_system.system import OpticalSystem
from optical_system.elements import Lens
from visualization.plotter import Plotter

wavelength = 0.8
x = np.linspace(-20, 20, 513)
X, Y = np.meshgrid(x, x)
U0 = np.exp(-(X**2 + Y**2) / 8.0**2).astype(np.complex128)

system = OpticalSystem(wavelength, x, x, U0)
system.add_element(Lens(z_position=50, focal_length=100))
results = system.propagate_to_cross_sections([0, 150])

plotter = Plotter(x, x)
plotter.plot_cross_sections(results, save_label='demo', show=False)
```

矢量版（同一条光路的偏振解析）：

```python
from vector.sources import vector_gaussian
from vector.elements import VectorLens
from vector.system import VectorOpticalSystem

src = vector_gaussian(x, x, wavelength, 4.0, 'radial')   # 径向偏振
vsys = VectorOpticalSystem(wavelength, x, x, src)
vsys.add_element(VectorLens(50, focal_length=100, NA=0.6))
E_focus = vsys.propagate_to_cross_sections([150])[150]   # 含 Ez 纵向场
```

## 文档导航

- [快速上手](quickstart.md) — 安装、第一个仿真、常见陷阱
- [角谱传播](physics/angular-spectrum.md) — 原理、采样定理、传播器缓存
- [多层膜 TMM 与传函耦合](physics/multilayer-tmm.md) — H(kx,ky) 口径与超透镜
- [矢量光学](physics/vector-optics.md) — 横场投影、RW 聚焦、q-plate
- [宽谱与参数扫描](physics/broadband-workflows.md) — C4/C3/A3 工作流层
- [API 参考](api/system.md) — 全部公共类
