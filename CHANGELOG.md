# Changelog

本项目的所有重要变更记录。格式参考 [Keep a Changelog](https://keepachangelog.com/)。

## [0.3.0] — 2026-09-06 (M1: 性能与耦合)

### Added
- **C1 缓存角谱传播器** `propagation/propagator.py::AngularSpectrumPropagator`：
  kz/网格一次性缓存，纵向多步扫描纯传播 **2.68×** 提速，
  与旧实现逐位一致（回归测试保证）；`OpticalSystem` 三处传播循环已接入。
- **D1 广义斜入射 TMM** `tmm_k2`（kx²+ky² 驱动）与 `tmm_k2_amplitudes`
  （复振幅 t/r）；掠入射精确临界 R→1 特判（消除除零 NaN）。
- **A2 `MultilayerSlab` 元件**：多层膜 TMM 场传函 H(kx,ky) 嵌入角谱传播——
  超透镜传函仿真正式进入框架。1D 径向采样 + GPU 插值；传播带 |H|²=T、
  倏逝带增强（实测 p 偏振 |H|max≈3.3 @365nm 结构）；验收测试链 7 项。
- **E1 CI**：GitHub Actions（ruff + py3.10/3.11 CPU 矩阵 + 可选自托管 GPU
  runner）；GPU 测试统一 `@pytest.mark.gpu` 标记体系。
- **E3 黄金图像回归**：dHash 感知哈希比对（阈值 6），防渲染/物理静默漂移。
- **E2 文档站**：mkdocs-material 骨架（快速上手/物理指南/API 参考/benchmark）。
- example `multilayer_slab_superlens.py`（365nm ZnO/Ag/ZnO 传函 + 近场演化）。

### Changed
- `multilayer` 纳入 editable 安装包（核心元件依赖 `multilayer.common.tmm`）。
- `requirements.txt` 补 nvidia-*-cu12 运行库轮子（GPU 免装 CUDA Toolkit）、
  补 `openpyxl`，移除未使用的 `colorama`/`scikit-image`。
- `.gitignore` 重写为 UTF-8；源数据（nk 表、MSPP npy）纳入版本控制。

### Fixed
-（继承自 v0.2.x 审计，见 AUDIT.md 与 tag v0.2.0）

### Performance
- 纯传播基准：513²×256 步 cached=0.179s vs legacy=0.479s（2.68×）。

## [0.2.0] — 2026-09-05 (全面审计修复)

- 修复 P0×5 / P1×8 / P2×6（包结构断裂、光阑构造崩溃、调试残留、GPU 环境、
  float64 默认精度、TMM 收敛到 `multilayer/common`、结构重组、工程化）。
- 完整报告见 [AUDIT.md](../AUDIT.md)。
