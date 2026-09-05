# 基准测试

环境：Windows 11，RTX 4060 8GB，Python 3.11.9，cupy-cuda12x 13.3.0，complex128。

## 传播器缓存（C1，v0.3.0）

513×513 网格、256 步累积传播（纯传播对比，等操作序列）：

| 实现 | 耗时 | 相对 |
|------|------|------|
| 旧 `angular_spectrum_propagate` 逐次调用 | 0.479 s | 1.00× |
| `AngularSpectrumPropagator.propagate` 缓存 | 0.179 s | **2.68×** |

提升来源：旧实现每步重建 fftfreq/meshgrid/KX/KY/kz 网格（8+ 次 kernel
launch + 全网格 sqrt）；缓存后每步只剩 `exp(1j·kz·z)` + FFT 对。
缓存构建（一次性）约 3 ms，纵向扫描步数越多收益越大。

回归：`tests/test_propagator.py::test_propagator_longitudinal_speedup`
（`slow` + `gpu` 标记，阈值防退化）。

## FFT 原始性能（历史数据，README 表）

| Framework | 256×256 | 1024×1024 | 8192×8192 |
|-----------|---------|-----------|-----------|
| NumPy (CPU) | 0.003 s | 0.038 s | 3.438 s |
| SciPy (CPU) | 0.001 s | 0.023 s | 1.919 s |
| CuPy (GPU) | 0.155 s | 0.018 s | 0.094 s |

注：CuPy 小尺寸耗时含首次 kernel 编译；基准脚本在 `examples/benchmark.py`。

## 已知显存预算（complex128）

| 网格 | 单场 | 典型峰值（传播中 ~5 场共存） |
|------|------|------|
| 2049² | 67 MB | ~350 MB |
| 4097² | 268 MB | ~1.4 GB |
| 8193² | 1.07 GB | ~5.5 GB（接近 8GB 上限）|
| 10241² | 1.68 GB | 超限 → 用 `dtype=cp.complex64` |
