# Changelog

本项目的所有重要变更记录。格式参考 [Keep a Changelog](https://keepachangelog.com/)。

## [0.5.0] — 2026-09-07 (M3: 宽谱/批量工作流)

### Added
- **C4 Field 结果容器** `workflows/field.py`：命名强度分量 + 可选复场
  快照 + 元数据；npz 压缩持久化（精确往返）；`line_cut` 截线；
  `from_scalar_system` / `from_vector_system` 一键提取（Field 是
  GPU → numpy 的数据边界）。
- **C3 ParameterSweep** `workflows/sweep.py`：参数网格笛卡尔积扫描
  执行器，`z_of` 钩子从参数推导传播平面；指标库（峰值/峰值坐标/
  窗口质心/FWHM 亚像素插值，解析高斯锚点验证 <1e-2）；结果直接进
  pandas / CSV，`best()` 按 nan 忽略择优。
- **A3 SimSpectrum** `workflows/spectrum.py`：宽谱照明的波长分解仿真
  ——非相干合成（Σw·|E|²，LED/荧光类）与相干合成（|Σ√w·E·e^{iφ}|²，
  相干长度覆盖光程差时）两种口径，权重自动归一，网格一致性校验。
- **桥接** `workflows/bridge.py::lift_to_vector`：标量场提升为指定
  偏振态 VectorField（近轴与标量传播一致 O(θ₀²)），矢量引擎续算入口。
- example `broadband_chromatic_shift.py`：衍射透镜色差焦移定量演示
  ——(λ, z) 二维扫描测得焦移曲线与理论 f₀λ₀/λ 吻合到网格分辨率，
  宽谱焦斑 FWHM 0.56→0.60μm、峰值 −25%（色差模糊）。
- 11 项 workflows 测试。

### Changed
- `pyproject.toml`：`workflows*` 纳入安装包，版本 0.5.0。
- `.gitignore` 补 `site/`（mkdocs 构建产物）。

## [0.4.0] — 2026-09-07 (M2: 全矢量引擎)

### Added
- **M2b 矢量角谱传播核心** `vector/propagator.py`：三分量场 (Ex,Ey,Ez) 的
  角谱传播——谱域 3D 横场投影 P = I − k̂k̂ᵀ 后逐谱点 exp(ikz·z) 相位，
  Ez 由投影重构。对任意输入处处有限（掠入射无除零奇异性）、投影幂等、
  功率 Parseval 守恒（1e-10）；近轴极限退化到标量传播器（O(θ₀²)）。
- **M2a 矢量光源** `vector/sources.py::vector_gaussian`：线偏振/±45°/
  LCP/RCP/径向/方位偏振高斯。
- **M2a Richards-Wolf 高NA焦场** `vector/richards_wolf.py`：Debye-Wolf
  积分的 FFT 实现（正弦条件光瞳、横场投影强度函数、1/kz 权重、全局
  相位与标量框架近轴对齐）；`focus_scan` 离焦扫描 Parseval 严格不变。
  验证：NA=0.85 径向/方位/圆偏振的纵向能量分数与解析积分一致
  （0.265 / 0.000 / 0.115）。
- **M2c Jones 偏振元件** `vector/elements.py`：WavePlate（QWP/HWP）、
  QPlate（自旋-轨道耦合，q=1/2 验证拓扑荷 ±1.00 精确）、Polarizer
  （Malus）、VectorAperture、VectorLens（标量 ObjectLens 同构矢量版）。
- **M2c VectorOpticalSystem** `vector/system.py`：元件排序编排 + 矢量
  传播，Jones 元件后零距离投影自动重构 Ez；横向/纵向截面输出。
- example `radial_polarization_focusing.py`（RW 与 VectorLens+ASP 双
  路径交叉对比，焦斑相关 0.958）、`qplate_vector_vortex.py`
  （径向对齐度 1.000000、传播后甜甜圈）。
- 24 项矢量引擎测试（tests/test_vector_*.py，gpu 标记）。

### Changed
- `pyproject.toml`：`vector*` 纳入安装包；版本号推进到 0.4.0
  （补 v0.3.0 时遗漏的版本号联动）。

### Fixed
- 高NA矢量传播的纵向场发散问题：横向迹直接做 Ez = −k⊥·A⊥/kz 重构在
  掠入射区（kz→0）发散；改为 3D 投影算子形式 Ez = −kz·D/k0² 后对所有
  谱点严格有限且满足 k·E = 0。

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
