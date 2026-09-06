# 工作流层 (workflows)

## 结果容器

::: workflows.field.Field
    options:
      members:
        - total_intensity
        - component
        - power
        - line_cut
        - save
        - load
        - from_scalar_system
        - from_vector_system

## 参数扫描与指标

::: workflows.sweep.ParameterSweep
    options:
      members:
        - run

::: workflows.sweep.evaluate_metrics

::: workflows.sweep.SweepResult
    options:
      members:
        - to_dataframe
        - best
        - save

## 宽谱合成

::: workflows.spectrum.SimSpectrum
    options:
      members:
        - run

::: workflows.spectrum.SpectrumResult
    options:
      members:
        - spectral_intensity

## 标量-矢量桥接

::: workflows.bridge.lift_to_vector
