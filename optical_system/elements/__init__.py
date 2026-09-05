"""光学元件包：显式导出全部元件类。

历史上元件导出依赖 ``elements_cls.py`` 末尾的循环 star-import，在命名空间包
（无 ``__init__.py``）下 ``from optical_system.elements import *`` 会导入 0 个
名字，导致所有实验脚本 NameError。现在统一在此显式导出。
"""

from optical_system.elements.apertures import (
    AnnularAperture,
    CircularAperture,
    CrossAperture,
    EllipticalAperture,
    RectangularAperture,
    SquareAperture,
)
from optical_system.elements.grating import (
    BlazedGrating,
    RectAmplitudeGrating,
    SineAmplitudeGrating,
    SinePhaseGrating,
)
from optical_system.elements.lens import Axicon, ObjectLens
from optical_system.elements.multilayer_slab import MultilayerSlab
from optical_system.elements.specific_elements import MSPP, SimpleMSPP
from optical_system.elements_cls import (
    Aperture,
    Lens,
    MomentumSpaceModulator,
    MomentumSpacePlate,
    OpticalElement,
    SpatialLightModulator,
    SpatialPlate,
)

__all__ = [
    # 基类
    "OpticalElement",
    "Aperture",
    # 空间元件
    "Lens",
    "ObjectLens",
    "SpatialPlate",
    "SpatialLightModulator",
    "Axicon",
    # 动量空间元件
    "MomentumSpacePlate",
    "MomentumSpaceModulator",
    "SimpleMSPP",
    "MSPP",
    # 光阑
    "CircularAperture",
    "SquareAperture",
    "EllipticalAperture",
    "RectangularAperture",
    "CrossAperture",
    "AnnularAperture",
    # 光栅
    "SinePhaseGrating",
    "RectAmplitudeGrating",
    "SineAmplitudeGrating",
    "BlazedGrating",
    # 多层膜
    "MultilayerSlab",
]
