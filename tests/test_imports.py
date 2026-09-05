"""导入链回归：保证 star-import 与全部元件导出可用。

历史 bug：无 __init__.py 的命名空间包 + elements_cls.py 末尾循环 star-import，
导致 from optical_system.elements import * 导出 0 个名字。
"""
import optical_system.elements as elements_pkg


def test_star_import_exports_all_elements():
    names = [n for n in dir(elements_pkg) if not n.startswith("_")]
    assert len(names) >= 20, f"元件导出数量异常: {names}"


def test_all_dunder_all_importable():
    for name in elements_pkg.__all__:
        assert getattr(elements_pkg, name) is not None


def test_all_apertures_importable():
    from optical_system.elements import (
        AnnularAperture,
        CircularAperture,
        CrossAperture,
        EllipticalAperture,
        RectangularAperture,
        SquareAperture,
    )
    for cls in (AnnularAperture, CircularAperture, CrossAperture,
                EllipticalAperture, RectangularAperture, SquareAperture):
        assert hasattr(cls, "create_mask")


def test_optical_system_importable():
    from optical_system.system import OpticalSystem
    assert hasattr(OpticalSystem, "propagate_to_cross_sections")
    assert hasattr(OpticalSystem, "propagate_to_longitudinal_section")
