"""M3 workflows：仿真工作流层——结果容器、参数扫描、宽谱合成。

模块结构：
- ``field``    : C4 Field 结果容器（元数据 + 强度/复场 + npz 持久化）；
- ``sweep``    : C3 ParameterSweep 参数扫描执行器 + 指标库（FWHM/质心）；
- ``spectrum`` : A3 SimSpectrum 宽谱仿真（非相干/相干合成）；
- ``bridge``   : 标量 → 矢量场桥接（矢量引擎续算入口）。
"""

from workflows.field import Field
from workflows.spectrum import SimSpectrum, SpectrumResult
from workflows.sweep import ParameterSweep, evaluate_metrics

__all__ = [
    "Field",
    "ParameterSweep",
    "evaluate_metrics",
    "SimSpectrum",
    "SpectrumResult",
]
