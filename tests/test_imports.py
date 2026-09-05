"""瀵煎叆閾惧洖褰掞細淇濊瘉 star-import 涓庡叏閮ㄥ厓浠跺鍑哄彲鐢ㄣ€?
鍘嗗彶 bug锛氭棤 __init__.py 鐨勫懡鍚嶇┖闂村寘 + elements_cls.py 鏈熬寰幆 star-import锛?瀵艰嚧 from optical_system.elements import * 瀵煎嚭 0 涓悕瀛椼€?"""
import optical_system.elements as elements_pkg


def test_star_import_exports_all_elements():
    names = [n for n in dir(elements_pkg) if not n.startswith("_")]
    assert len(names) >= 20, f"鍏冧欢瀵煎嚭鏁伴噺寮傚父: {names}"


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
