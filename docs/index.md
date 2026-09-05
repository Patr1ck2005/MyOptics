# MyOptics

MyOptics 是一个 GPU 加速的傅里叶光学仿真框架，核心是角谱法（Angular Spectrum
Method）传播器与可组合的光学元件系统，并将多层膜 TMM 传递函数作为动量空间
元件嵌入光路（超透镜传函仿真）。

## 核心能力

- **角谱传播**：严格（含倏逝波）/ Fresnel 双模式，缓存传播器
  （`AngularSpectrumPropagator`），纵向多步扫描提速 ~2.7×
- **元件库**：透镜（近轴/非近轴高 NA）、光阑（6 种）、光栅（4 种）、
  轴棱锥、动量空间调制器（MSPP）
- **多层膜耦合**：`MultilayerSlab` 把 TMM 场传函 H(kx,ky) 作用于光场，
  传播带透过率与倏逝增强（超透镜机制）一次到位
- **可视化**：横截面 / 纵截面 / 动量空间，LogNorm 阈值显示

## 安装

```bash
pip install -r requirements.txt
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

## 文档导航

- [快速上手](quickstart.md) — 安装、第一个仿真、常见陷阱
- [角谱传播](physics/angular-spectrum.md) — 原理、采样定理、传播器缓存
- [多层膜 TMM 与传函耦合](physics/multilayer-tmm.md) — H(kx,ky) 口径与超透镜
- [API 参考](api/system.md) — 全部公共类
