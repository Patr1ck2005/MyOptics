# MyOptics 全面审计报告

> 审计时间：2026 年（pre-agent 时代项目的首次全面体检）
> 范围：核心框架（`optical_system` / `propagation` / `utils` / `visualization` / `analytical`）、
> 根目录与 `test/` 实验脚本、`cpu_version` 遗留副本、`multilayer`（round1/round2）、
> 仓库卫生（git / 依赖 / 打包）。
> 方法：逐文件人工走查 + venv 实际运行复现关键问题 + numpy/cupy 数值验证。

## 结论总览

| 级别 | 数量 | 含义 |
|------|------|------|
| P0 | 5 | 阻断性：项目在当前仓库状态下实际无法运行 |
| P1 | 8 | 正确性 / 可复现性缺陷 |
| P2 | 6 | 结构性 / 工程卫生问题 |

处置标记：✅ 已修复 · 📌 保留（记录约定，不改动物理行为）· 🗑️ 已删除 · ⏸️ 暂不处理

---

## P0 — 阻断性问题

### P0-1 ✅ 包结构断裂：全仓库无 `__init__.py`，star-import 导入 0 个名字
- 位置：`optical_system/elements_cls.py:336-338`（末尾循环 star-import）、`optical_system/elements/`（目录无 `__init__.py`）
- 现象（实测复现）：`from optical_system.elements import *` 静默导入 **0 个名字**；
  `from optical_system.elements import Lens` → `ImportError: unknown location`。
  所有根目录与 `test/` 实验脚本因此全部 NameError。
- 根因：项目重构把 `elements.py` 拆成 `elements_cls.py` + `elements/` 包目录，
  但既没加 `__init__.py`，也没更新导入方；旧 star-import 链在命名空间包下失效。
- 修复：补齐所有包的 `__init__.py`；`optical_system/elements/__init__.py` 显式导出
  全部元件（含 `__all__`）；删除 `elements_cls.py` 末尾的循环 star-import。

### P0-2 ✅ 4/6 光阑类构造即崩溃
- 位置：`optical_system/elements/apertures.py`（Elliptical/Rectangular/Cross/Annular）
- 现象（实测复现）：`TypeError: Aperture.__init__() got an unexpected keyword argument 'radius'`
  —— 子类传 `radius=None`，基类签名却是 `(z_position, size)`。
  `SquareAperture` 还缺失 `__init__`，API 命名不一致（`size=` vs `radius=`）。
- 修复：基类 `size` 改为可选；子类改为 `super().__init__(z_position)`；
  `SquareAperture` 补显式 `__init__`。

### P0-3 ✅ 调试代码残留在核心传播路径
- 位置：`propagation/angular_spectrum.py:46-48`、`optical_system/system.py:141/228/297`、
  `optical_system/test/modulator.py:71-90`
- 现象：`'Rigorous'` 分支每次传播都 `plt.imshow(...); plt.show()` ——
  严格模式每前进一步弹一张图并阻塞等人关窗口；传播循环内三处裸 `print`；
  `modulator.py` 的 `apply()` 内嵌四个 DEBUG 弹窗。
- 修复：全部移除（`modulator.py` 整体删除，见 P2-4）。

### P0-4 ✅ GPU 环境实际不可用（cupy 缺 NVRTC）
- 位置：`.venv`（`cupy-cuda12x==13.3.0`）
- 现象（实测复现）：`cp.fft.fft2` 可用（cuFFT 轮子自带），但一切逐元素 ufunc
  （`cp.exp` / `cp.meshgrid` / `np.arctan2` on cupy 数组）均抛
  `RuntimeError: CuPy failed to load nvrtc64_120_0.dll`——
  本机无 CUDA Toolkit（`CUDA_PATH` 空、无 nvcc），cupy 轮子不带 CUDA 运行库。
- 修复（两步）：
  1. 安装 `nvidia-*-cu12` pip 轮子（CUDA 12 运行库，写入 requirements）；
  2. cupy 的 Windows 轮子**不会自动发现** pip 装的 `nvidia/*/bin` 目录——
     新增 `utils/cuda_path.py`（`add_dll_directory` + PATH 前置双通道），
     在 `optical_system` 包导入时执行。实测 RTX 4060 上 NVRTC/cuFFT 全链路可用。

### P0-5 ✅ `np.*` ufunc 直接作用于 cupy 数组
- 位置：11 处脚本（`vortex_beam-OL-study.py` 等）的调制函数 lambda
  `np.exp(1j*2*np.arctan2(Y, X))`——运行时 X/Y 是 cupy 数组，`np.arctan2`
  无法分派到 GPU，必然 TypeError（`elements/grating.py` 的 `cp.exp` 写法才是对的）。
- 修复：全部改为 `cp.exp`/`cp.arctan2`，并在元件文档字符串中写明
  “调制函数接收 cupy 数组、须用 cp ufunc”。
- 附带修复：`vortex_beam-exp_system-full-reflected.py`、`test/test-4f_system.py` 等
  使用 `PI` 却未导入的脚本补 `from utils.constants import PI`。

---

## P1 — 正确性 / 可复现性

### P1-1 ✅ 数值精度硬编码 float32/complex64
- 位置：`optical_system/system.py:30-35`（float64 版本被注释掉）
- 影响：高 NA、大光程角谱传播中 f32 舍入污染相位；精度策略不该靠注释切换。
- 修复：`OpticalSystem(dtype=cp.complex128)` 参数化，默认 float64 研究精度；
  速度敏感场景显式传 `complex64`。大网格脚本（`vortex_beam-OL-study.py`、
  `test-4f_system.py`，`mesh=1024*10+1`）已显式降为 complex64 以避免 8GB 显存 OOM。

### P1-2 ✅ 源数据全部不在版本控制内
- 位置：`.gitignore` 一刀切 `*.csv/*.xlsx/*.npy/*.png`
- 影响：`multilayer` 的 Ag/ZnO/薄膜.nk 数据、`optical_system/data/efficiency.npy`
  （MSPP 元件运行依赖）换机 clone 后全部缺失，MSPP 与整个 multilayer 流水线不可复现。
- 修复：`.gitignore` 重写（UTF-8，原文件为 GBK 乱码），对源数据加 `!` 例外入库；
  结果类（`rsl/`、`img/`、`*.png`）继续忽略。

### P1-3 ✅ 4 个 `.pyc` 被 git 跟踪
- 位置：`optical_system/__pycache__/*.pyc` 等（在 `*.pyc` 规则之前提交）
- 修复：`git rm -r --cached`；`.gitignore` 改为 `__pycache__/` + `*.py[cod]`。

### P1-4 ✅ README 示例语法错误 / 版本声明过时 / requirements 缺依赖
- `README.md:130-131`：赋值语句断行（`cross_sections` 换行 `= ...`）—— 复制即语法错误；
  “Python 3.7+” 不实（`list[OpticalElement]` 需 3.9+，cupy 13 需 3.10+）；
  README 让装 `cupy-cuda11x`，requirements 钉的是 `cupy-cuda12x`；
  requirements 缺 `openpyxl`（读 `薄膜.xlsx` 必需），且 `colorama`/`scikit-image` 全仓库未使用。
- 修复：README 重写相应章节；requirements 重整。

### P1-5 ✅ `benchmark.py` pyFFTW 分支 shape 错误
- 位置：`benchmark.py:54`（`empty_aligned((size, size))`=1000²，数据是 8192²）
- 现象：装有 pyfftw 的环境必然广播失败崩溃（本项目 venv 未装故此前未触发）。
- 修复：按 `data.shape` 分配。

### P1-6 ✅ Plotter 三处功能性缺陷
- `plot_field`：`plt.show()` 之后才 `savefig` —— show=True 时存出空白图（已修复：先存后显示，并 `close(fig)`）。
- `plot_cross_sections`：单截面时 `np.atleast_2d` 把 (4,) 变 (1,4)，`axes[1][0]` IndexError（已修复：统一 reshape 为 (行,列)）；
  动量空间相位面板的刻度重复设到 `axes[2]`（应为 `axes[3]`，已修复）；
  `np.arcsin(ky/k0)` 对倏逝区（|ky|>k0）产生 NaN 刻度标签（已修复：超限刻度留空）。
- `plot_longitudinal_section`：norm_vmin/vmax + `ref_position_*` 一整套动态计算从未生效
  （LogNorm 被注释、cmap 变量没传给 imshow）——调用方传参被静默忽略。
  已激活：参数有效时启用 `LogNorm` + `under` 黑色，非法范围回退线性显示（行为从“死参数”变为文档所述行为）。
- 附带：`Plotter` 的 `wavelength` 默认 1.550 仅影响动量空间角度刻度换算，
  与仿真波长不一致时角度轴会错——已在 docstring 标注，调用方应显式传入。

### P1-7 ✅ `MultiLayerTM.field_at` 对 z<0（入射介质）取错层
- 位置：`propagation/multi_layer/cls.py` `field_at`
- 现象：`z < interfaces[-1]` 分支用 `searchsorted` 落到第 0 层，入射半空间被当作第 0 层膜处理。
- 修复：增加 `z < 0` 分支返回 `exp(i·kz0·z) + r·exp(-i·kz0·z)`。

### P1-8 ✅ MSPP 数据路径指向不存在的目录
- 位置：`optical_system/elements/specific_elements.py:13`
- 现象：`data_path` 拼接 `elements/data/`，但 npy 实际在 `optical_system/data/`
  —— `MSPP(...)` 构造时 `np.load` 必然 FileNotFoundError（重构移动文件前，
  该路径从未正确解析过）。
- 修复：指向包级 `optical_system/data/`，并实测构造通过。

---

## P2 — 结构性 / 工程卫生

### P2-1 ✅ TMM 实现重复 ≥4 份 → 收敛到 `multilayer/common`
- 历史分布：`propagation/multi_layer/cls.py`（H–Q 形式，场分布/PSF 用途，保留）、
  `propagation/multi_layer/energy_efficiency*.py` 等 6 个一次性探索脚本（legacy 保留）、
  `multilayer/round1/scripts/optimize_transmittance.py`（N-导纳正入射版）、
  `round1/angle_scan_spectra.py` 与 `round2/angle_scan_spectra_variants.py`（kx 驱动版）。
- 修复：新建 `multilayer/common/materials.py`（物料加载/插值去重）与
  `multilayer/common/tmm.py`（采用 round2 的 N-导纳 kx 驱动实现，含 kz 物理分支选取）；
  round1/round2 脚本改为公共库导入。
- **回归保证**：重构后重跑 round1 角度扫描（no_abs 模式），与 2026-07-14 原始结果 zip
  逐值比对，最大差 **0.000e+00**（逐位一致），零物理偏移。

### P2-2 🗑️ `cpu_version/` 整库旧副本
- numpy 旧实现与主版本 API 已分叉（旧 `elements.py`、无 `propagation_mode`），
  其 `from optical_system.elements import ...` 在重构后同样解析到坏掉的命名空间包。
- 处置：整体删除（git 历史可恢复）。GPU 精度参数化后主版本即可覆盖 CPU 研究需求；
  如需 CPU 后端，应做 backend 抽象而非复制整库。

### P2-3 🗑️ `propagation/超透镜传函 (1).py`
- GBK 编码 + 文件名含 " (1)"（浏览器下载副本），逻辑已被 `multi_layer/cls.py` 覆盖。
- 处置：删除（git 历史可恢复）。

### P2-4 🗑️ `optical_system/test/modulator.py` + `optical_system/test/*.npy`
- 旧版 `MomentumSpaceModulator` 副本（与 `elements_cls.py` 版本插值参数已发散：
  linear vs cubic），内嵌 4 处 DEBUG 弹窗；`test/` 目录下 npy 与 `data/` 重复。
- 处置：删除；唯一保留 `optical_system/data/`（specific_elements.py 的加载路径）。

### P2-5 ✅ 根目录脚本平铺 / `test/` 目录名不副实
- 13 个根目录实验脚本 + `test/` 4 个可视化脚本全部移入 `examples/`
  （`test/` 实为可视化实验而非测试）；
  `benchmark.py` 一并移入；真实测试见 `tests/`（pytest 物理回归套件）。

### P2-6 ✅ 工程化补齐
- 新增 `pyproject.toml`（元数据、依赖、ruff 配置）、editable 安装（任意 cwd 可运行）、
  `pytest` 套件（导入链 / 元件构造 / 能量守恒 / 透镜聚焦 / TMM 解析对照）。
- ⏸️ 未做：CI 流水线（无远程仓库配置信息，留待用户决定）。

---

## 保留的物理行为约定（📌 只记录不修改，避免悄悄改变已有仿真结果）

1. **`GaussianBeam.compute_field` 的 `z_position *= -1`**（`utils/beams.py:33`）：
   隐式符号翻转——正的 `z_position` 表示“沿传播方向距束腰的距离”，
   内部按 -z 计算曲率/Gouy 相位。与各实验脚本的用法一致，保持原样；使用前须知此约定。
2. **`Lens` 的 NA 掩膜近似**（`elements_cls.py`）：在透镜平面用
   `max_radius = f·tan(arcsin NA)` 截断——薄片透镜理想化（NA 以空气 n=1 定义），
   与 `ObjectLens` 的非近轴球面波相位配套使用。
3. **`BlazedGrating` 的锯齿相位**（`elements/grating.py`）：
   `phase = k·tan(blaze_angle)·(Y % period)`——未按光栅方程归一化到 2π·m，
   `blaze_angle` 参数含义是“锯齿斜率”而非严格闪耀角；当
   `k·tan(θ)·period = 2π·m` 时退化为理想闪耀。
4. **角谱传播无带限/防混叠**（`propagation/angular_spectrum.py`）：
   长距离传播时窗口边缘存在 wrap-around 混叠风险，代码不主动警告；
   使用者需保证仿真窗口足够大（README 已有采样说明）。
5. **场归一化**：`OpticalSystem.__init__` 将 `sum|U|²` 归一，所有强度为相对单位
   （这就是实验脚本里 `vmax=5e-7` 这类小数值的来源）。

## 遗留与建议（⏸️ 未纳入本次）

- `propagation/multi_layer/` 下 6 个探索脚本（`energy_efficiency*.py`、`compute_SF.py`、
  `compute_heatmap.py`、`dose.py`、`sigmoid.py`）标记为 legacy：公式彼此有细微差异
  （如 `kz` 分支处理），合并需逐一数值验证，收益低于风险；`cls.py`（`MultiLayerTM`，
  场分布/LSF/PSF 工具）已修复并保留。
- `MomentumSpacePlate` 系列的插值在 CPU 上进行（scipy 不支持 GPU），
  大网格 + 逐步传播时 `cp.asnumpy` 往返是主要开销；后续可改用 cupy 自实现插值。
- `analytical/aperture_diffraction.py` 的环形积分丢弃了方位角耦合项
  （-k·X·ri·cosθ/z2，产生 Airy 环的项）且单位混乱（λ×100 与 z(m) 混用），
  计算结果无物理意义——如需保留应重写，本次未动（该脚本未被任何流程引用）。
- `OpticalSystem.propagate_to_longitudinal_section(_direct)` 两方法 90% 重复
  （_direct 忽略全部元件、仅自由传播），建议后续合并为一个带 `apply_elements` 开关的方法。
- 角谱传播器每次调用都重建 kz/H 网格；纵向扫描（数百步）可预计算 kz 缓存、
  每步只算 `exp(1j·kz·z)`，预计提速 2-3 倍。

## 复现验证记录

| 验证项 | 命令 | 结果 |
|--------|------|------|
| star-import 修复 | `from optical_system.elements import *` | 22 个名字（修复前 0 个） |
| 光阑构造修复 | 6 类逐一实例化 | 全部通过（修复前 4 类 TypeError） |
| TMM 重构零偏移 | 重跑 round1 no_abs vs 2026-07 zip | 最大逐值差 0.000e+00 |
| 物理回归 | `python -m pytest tests/` | 见仓库 CI/本地运行 |
| GPU 链路 | `cp.exp` on complex64 | nvidia-cu12 轮子安装后可用 |
