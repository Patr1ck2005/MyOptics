"""M2 全矢量引擎：矢量光场、矢量角谱传播、Richards-Wolf 高NA聚焦、Jones 偏振元件。

模块结构：
- ``field``        : VectorField 三分量复场容器；
- ``propagator``   : 矢量角谱传播器（3D 横场投影 P = I − k̂k̂ᵀ + Ez 重构）；
- ``sources``      : 矢量高斯光源（线/圆/径向/方位偏振）；
- ``richards_wolf``: aplanatic 光瞳 + 高NA矢量焦场（Debye-Wolf 积分）；
- ``elements``     : Jones 偏振元件（波片/q-plate/偏振片/光阑/高NA透镜）；
- ``system``       : VectorOpticalSystem 端到端矢量仿真。
"""

from vector.elements import (
    HalfWavePlate,
    Polarizer,
    QPlate,
    QuarterWavePlate,
    VectorAperture,
    VectorLens,
    WavePlate,
)
from vector.field import VectorField
from vector.propagator import VectorAngularSpectrumPropagator
from vector.richards_wolf import RichardsWolfFocuser
from vector.sources import vector_gaussian
from vector.system import VectorOpticalSystem

__all__ = [
    "VectorField",
    "VectorAngularSpectrumPropagator",
    "RichardsWolfFocuser",
    "VectorOpticalSystem",
    "vector_gaussian",
    "WavePlate",
    "QuarterWavePlate",
    "HalfWavePlate",
    "QPlate",
    "Polarizer",
    "VectorAperture",
    "VectorLens",
]
