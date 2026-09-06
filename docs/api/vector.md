# 矢量引擎 (vector)

## 场与传播

::: vector.field.VectorField
    options:
      members:
        - intensity
        - power
        - normalized
        - copy

::: vector.propagator.VectorAngularSpectrumPropagator
    options:
      members:
        - project
        - propagate
        - propagate_batch

## 光源与高 NA 聚焦

::: vector.sources.vector_gaussian

::: vector.richards_wolf.RichardsWolfFocuser
    options:
      members:
        - focus
        - focus_scan

## 偏振元件

::: vector.elements.WavePlate

::: vector.elements.QuarterWavePlate

::: vector.elements.HalfWavePlate

::: vector.elements.QPlate

::: vector.elements.Polarizer

::: vector.elements.VectorAperture

::: vector.elements.VectorLens

## 系统

::: vector.system.VectorOpticalSystem
    options:
      members:
        - add_element
        - propagate_to_cross_sections
        - propagate_to_longitudinal
