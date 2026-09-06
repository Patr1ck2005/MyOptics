# 宽谱与参数扫描（M3 工作流层）

`workflows/` 包提供三层复用工作流：结果容器（C4）、参数扫描（C3）、
宽谱合成（A3）。它们消费 `OpticalSystem` / `VectorOpticalSystem` 的
输出，把「跑一次仿真」升级为「跑一族仿真 + 标准化产出」。

## C4 Field：结果的标准容器

```python
from workflows.field import Field

fields = Field.from_scalar_system(system, z_positions=[10.0])
f = fields[0]                 # 一次仿真平面截面的快照（numpy）
f.intensities                 # {'U': ...}；矢量仿真为 {'ex','ey','ez'}
f.total_intensity             # 各分量之和
f.power()                     # Σ I · dx · dy
f.line_cut(axis='x')          # 过峰值截线
f.save('result.npz')          # 压缩持久化（元数据 JSON 编码）
f2 = Field.load('result.npz') # 精确往返
```

Field 是**数据边界**：GPU 张量在入口转 numpy，下游分析/落盘/传输
不再接触 cupy。`keep_complex=False` 可丢弃复场快照省内存。

## C3 ParameterSweep：参数扫描 + 指标

```python
from workflows.sweep import ParameterSweep

sweep = ParameterSweep(
    build_system=lambda wavelength: build_my_system(wavelength),
    metrics=('peak_intensity', 'fwhm_x', 'fwhm_y', 'centroid'),
)
result = sweep.run({'wavelength': [0.4, 0.5, 0.6]},
                   z_of=lambda wavelength: 14.0)   # 传播平面（可随参数变化）
df = result.to_dataframe()      # pandas：参数列 + 指标列
result.best('fwhm_x', mode='min')
result.save('sweep.csv')
```

要点：

- `build_system` **每次调用返回全新系统**（系统有内部状态）；
- 二维 `(λ, z)` 网格扫描时用 `z_of=lambda wavelength, z: z` 把传播
  平面从参数推导出来（参数全集都会传给 build_system，构建时忽略
  不需要的键）；
- 指标 FWHM 用过峰值截线 + 线性插值亚像素定位，解析高斯锚点验证
  （强度 I = exp(−r²/σ²) 的 FWHM = 2√(ln2)·σ）到 <1%；
- 自定义指标传 `evaluate=lambda field, **params: {...}`。

## A3 SimSpectrum：宽谱照明的波长分解

宽谱场按波长分解为一系列独立单色仿真，再按光谱权重合成：

- **非相干合成**（默认）：I_total = Σ_s w_s·I_s。适用条件：宽谱
  相干长度 ≪ 系统最大光程差（LED、荧光、太阳光）——各波长分量
  互不相干，强度直接相加；
- **相干合成**：E_total = Σ_s √w_s·E_s·e^{iφ_s}。适用条件：各分量
  光程差 < 相干长度（多纵模激光、超短脉冲载波近似）。跨波长相位
  φ_s 的物理有效性取决于系统光程建模精度，需自行判断。

```python
from workflows.spectrum import SimSpectrum

spec = SimSpectrum(lambdas, weights=spectrum_power, simulate=simulate_at)
result = spec.run(coherent=False)
result.field                 # 合成后的 Field（wavelength=None）
result.components            # {λ: 单色 Field}
result.spectral_intensity    # 各波长的功率传输
```

!!! warning "色差物理"
    宽谱通过**色散元件**时，各波长的焦面不同（衍射透镜
    f(λ) = f₀λ₀/λ，阿贝数 ≈ −3.45，比玻璃强一个量级）。非相干合成
    在单一平面得到的是「各波长最佳焦斑的加权叠加」——焦移未补偿时
    表现为 FWHM 恶化 + 峰值下降。见
    `examples/broadband_chromatic_shift.py` 的定量演示。

## 桥接：标量 → 矢量

`workflows.bridge.lift_to_vector(U, x, y, λ, polarization)` 把标量场
提升为指定偏振态（x/y/±45/LCP/RCP/径向/方位）的 `VectorField`，
供矢量引擎续算（偏振光学、纵向场）。近轴下与标量传播一致到 O(θ₀²)。
