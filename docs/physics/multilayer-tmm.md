# 多层膜 TMM 与传函耦合

## 两套 TMM 的分工

| 模块 | 形式 | 用途 |
|------|------|------|
| `multilayer/common/tmm.py` | N-导纳形式，kx/kx² 驱动 | 角度谱 T/R 求解、传函来源 |
| `propagation/multi_layer/cls.py` | H–Q 形式（`MultiLayerTM`） | 场分布 H(z)、LSF/PSF |

公共约定：长度 nm、波长 μm、膜堆顺序 入射介质 → 有限层 → 出射介质。
`kz` 物理分支选取规则：$\mathrm{Im}(k_z)\ge 0$；近实根取 $\mathrm{Re}(k_z)\ge 0$。

## 场传函 H(kx, ky)

`MultilayerSlab` 把膜堆作为**动量空间滤波器**嵌入光路：

$$U_{out}(x,y) = \mathcal{F}^{-1}\{\, H(k_x,k_y)\, \mathcal{F}[U_{in}]\,\}$$

其中 $H$ 是 TMM 的**场振幅比** $t_{amp}$（透射侧出射场 / 入射侧场，含相位）：

- **传播带**（$k<k_0$）：$|H|^2$ 即功率透过率 $T(k)$，能流口径自洽；
- **倏逝带**（$k>k_0$）：$|H|$ 可以 **> 1** —— 这不是能量守恒破坏，
  倏逝分量不携带净能流，金属膜表面的近场增强（超透镜机制）正来自这里。
  实测 ZnO/Ag/ZnO @365nm p 偏振 $|H|_{max}\approx 3.3$。

!!! warning "口径约定（不要"修正"它）"
    H **不做** $|H(0)|=1$ 归一、**不做**能流归一。归一会抹掉倏逝增强，
    超透镜仿真立即失真。需要绝对效率时查 `peak_transmittance` 属性
    （$=|H(0)|^2$）。

## 实现要点

- **径向压缩**：膜堆 z 轴对称 → $H$ 只依赖 $k_r=|\mathbf{k}|$。
  1D 采样 $n_{kr}$ 点（默认 4096，kr 非均匀加密：光锥边界与深倏逝两端密），
  GPU 线性插值到 2D 网格。成本 $O(n_{kr})$ 次 TMM，与网格无关。
- **插值精度**：含金属膜堆在传播带内就有类 F-P 快变结构（尺度
  ~$10^{-4}\ \mathrm{rad/nm}$），采样密度直接决定精度；4096 点对应
  ~$10^{-4}$ 相对误差。出现谐振锐峰时调大 `n_kr`。
- **fftshift 构造**：2D 传函在 fftshift 排序（单调、DC 居中）的频率轴上
  构造后 `ifftshift` 回 fftfreq 顺序，规避奇数网格 Nyquist 陷阱。

## 一致性要求：空气包覆

框架的自由传播按真空 $k_0$ 展开。`MultilayerSlab` 默认 `N_inc=N_exit=1`
（空气）时与前后传播严格自洽；`N_exit ≠ 1` 的出射介质内传播相位/衰减
不被框架表达（需用 `MultiLayerTM.field_profile` 单独分析）。

## 验收测试链（tests/test_multilayer_slab.py）

1. **退化**：空气板 ≡ 标准角谱传播（3.7e-6）；
2. **对称**：$H(k_x,k_y)$ 径向对称；
3. **交叉**：$H(k_r)$ 与 `tmm_k2_amplitudes` 逐点一致；
4. **物理**：倏逝增强存在（$|H|>1$）；传播带 $|H|^2 = T$；零厚度恒等。
