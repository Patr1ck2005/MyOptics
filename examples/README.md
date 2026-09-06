# Examples

从仓库根目录运行（需要先 `pip install -e .` 并配好 GPU 环境）。

## 受支持示例（随版本维护，测试覆盖物理结论）

| 脚本 | 演示内容 | 关键结论 |
|------|----------|----------|
| `multilayer_slab_superlens.py` | ZnO/Ag/ZnO 膜堆作为动量空间滤波器（超透镜传函） | 365nm p 偏振倏逝增强，近场演化 |
| `radial_polarization_focusing.py` | 径向偏振高 NA 聚焦（RW 与 VectorLens 双路径） | Ez 焦斑中心主导；两模型 NA=0.85 相关 0.96（模型差见面板注记） |
| `qplate_vector_vortex.py` | q-plate 自旋-轨道耦合 → 矢量涡旋 | 径向对齐度 1.000000，拓扑荷 −1.00，传播后甜甜圈 |
| `broadband_chromatic_shift.py` | 衍射透镜色差焦移（SimSpectrum + ParameterSweep） | 焦移曲线吻合 f₀λ₀/λ 理论；宽谱焦斑 FWHM 0.56→0.60μm |
| `benchmark.py` | FFT 性能基准（MATLAB/NumPy/SciPy/CuPy 对照数据来源） | 见 `docs/benchmark.md` |

## legacy/（历史探索脚本，未随版本验证）

v0.2 审计前的研究脚本（vortex_beam 系列实验、grating vortex、4f 系统
试验等），保留作研究记录。它们使用旧 API 惯例（星号导入、`Plotter`
直连、`GaussianBeam` 的 z 符号翻转约定），**不保证在当前版本运行**；
需要复用其中思路时，建议按 `docs/quickstart.md` 的现行 API 重写。
