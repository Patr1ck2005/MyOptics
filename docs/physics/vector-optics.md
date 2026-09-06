# 矢量光学：全矢量引擎（M2）

标量角谱传播把光场当作一个复标量 U(x,y)，隐含假设偏振结构在传播中
不变。这对近轴小 NA 场是好的近似；但高 NA 聚焦、偏振元件、矢量光束
（径向/方位偏振、矢量涡旋）都必须回到麦克斯韦方程的矢量本质。
本框架的矢量引擎（`vector/` 包，v0.4.0）在角谱框架内严格处理三分量
场 **E = (Ex, Ey, Ez)**。

## 自由空间中的矢量场：横场条件

时谐约定 e^{i(k·r − ωt)} 下，真空中的每个平面波分量必须满足横场条件：

$$
\mathbf{k} \cdot \mathbf{E}(\mathbf{k}) = 0
$$

关键观察：**任意横向迹 (Ex, Ey) 并不自动是自由空间麦克斯韦解**——
它一般含 k̂ 方向的非物理分量。矢量引擎的处理是谱域 3D 横场投影：

$$
\hat{P} = I - \hat{k}\hat{k}^{\mathsf T}, \qquad
\begin{pmatrix} A_{ex}' \\ A_{ey}' \\ A_{ez}' \end{pmatrix}
= \hat{P} \begin{pmatrix} A_{ex} \\ A_{ey} \\ 0 \end{pmatrix}
= \begin{pmatrix}
A_{ex} - k_x D/k_0^2 \\
A_{ey} - k_y D/k_0^2 \\
-\,k_z D/k_0^2
\end{pmatrix}, \quad
D = k_x A_{ex} + k_y A_{ey}
$$

投影后逐谱点乘传播相位 exp(i k_z z)（k_z 取 Im ≥ 0 分支，倏逝波沿 +z
衰减）。该形式有三个好的性质：

- **处处有限**：Ez′ ∝ k_z，掠入射（k_z → 0）时 Ez′ → 0，无除零奇异性；
  倏逝区 k_z = iγ 自然给出指数衰减的纵向场；
- **幂等**：输入已是横场时 D₃ = k·A = 0，投影是恒等操作——元件链中
  反复投影不累积误差；
- **守恒**：投影后谱域逐点纯相位，Σ|E|² 网格和严格 Parseval 守恒。

近轴极限下投影是 O(θ₀²) 修正，横向分量退化到标量传播器（测试保证
一致到 5e-3）；差别本身是物理——"横向迹 → 自由场解"必须剔除 k̂ 分量。

## Richards-Wolf 高 NA 矢量焦场

高 NA 物镜把入瞳场映射到出瞳方向谱 (kx, ky)，焦场是 Debye-Wolf 积分：

$$
\mathbf{E}(\mathbf{r}) = C \iint_{k_\perp \le k_0 \mathrm{NA}}
\frac{\mathbf{E}_\infty(k_x,k_y)}{k_z}
\, e^{-i(k_x x + k_y y)} \, e^{+i k_z z} \, dk_x dk_y
$$

- **光瞳映射**（正弦条件/aplanatic）：ρ = f·sinθ，sinθ = k⊥/k0；
- **强度函数**：入瞳场 E_in 折射到方向 k̂ 后投影回横场平面，
  E∞ = √cosθ·[E_in − k̂(k̂·E_in)]，√cosθ 为能量 apodization；
- **权重 1/kz** 来自立体角换算 dΩ = dkx dky/(k0 kz)；
- **全局相位** C = (f/2π)·e^{i(k0 f − π/2)}，由近轴极限与标量框架
  （ObjectLens + 角谱传播）严格对齐定出。

标志性物理（NA=0.85，见 `examples/radial_polarization_focusing.py`）：

| 入射偏振 | Ez 能量分数（解析） | 焦斑形态 |
|---|---|---|
| 径向 radial | ≈ 26.5%（场积分口径） | 中心亮斑，轴上横向场对称归零 |
| 方位 azimuthal | ≈ 0 | 中空暗核环 |
| 圆偏振 LCP/RCP | ≈ 11.5% | 中心亮斑 + 偏振扭结 |

!!! note "两种高NA模型"
    框架提供两条高 NA 路径：`RichardsWolfFocuser`（经典 Debye 近似，
    光瞳坐标 f·sinθ，谱权重几何化）与 `VectorLens` + 矢量角谱传播
    （薄透镜相位 + 严格传播，光瞳坐标 f·tanθ，与标量 ObjectLens
    同构）。近轴极限下二者一致；高 NA 下的差异是**模型差异**而非
    数值误差（NA=0.85 时焦斑相关系数 ~0.96，见 example）。

## Jones 偏振元件与 q-plate

薄偏振元件作用在横向 Jones 矢量 (Ex, Ey) 上（2×2 矩阵逐点乘法），
输出 Ez=None，由下一次传播从横场条件重构。约定（`vector/elements.py`
模块 docstring 有完整推导）：

- 时谐 e^{−iωt} 基矢 (x̂, ŷ)，迎着 +z 传播方向看逆时针为 LCP，
  |L⟩ = (1, +i)/√2，|R⟩ = (1, −i)/√2；
- 线延迟器（快轴角 α，延迟 δ>0）快轴本征值 e^{−iδ/2}：
  J = R(α)·diag(e^{−iδ/2}, e^{+iδ/2})·R(−α)；
- 验证锚点：QWP(45°) 把 x̂ → RCP；HWP 把 |L⟩ → e^{2iα}|R⟩
  （手性翻转 + 几何相位 2α）。

**q-plate**（快轴 α(φ) = qφ + α0 的空间变向半波片）实现自旋-轨道耦合：

$$
|L\rangle \xrightarrow{\ \mathrm{q\text{-}plate}\ } e^{+2i\alpha(\phi)} |R\rangle,
\qquad
|R\rangle \rightarrow e^{-2i\alpha(\phi)} |L\rangle
$$

圆偏振高斯输入 → 相反手性、拓扑荷 ±2q 的标量涡旋；线偏振输入
（|L⟩+|R⟩ 叠加）→ 径向/方位型**矢量涡旋**。框架验证（q=1/2）：
RCP 输出的环路相位缠绕 = −1.00（精确），x 输入的局部偏振与 r̂ 的
归一化内积处处 = 1（允许逐点几何相位）。见
`examples/qplate_vector_vortex.py`。

## VectorOpticalSystem

与标量 `OpticalSystem` 相同的编排模式：元件按 z_position 排序，传播到
元件位置应用元件，输出各目标平面的完整三分量场。区别：

- 场容器是 `VectorField`（ex, ey, ez 可为 None）；
- 传播器每步做横场投影 + Ez 重构；
- Jones 薄元件作用后零距离 `project()` 保证任何输出平面带完整三分量；
- 功率约定与标量框架一致（初始化归一 Σ|E|²·dx·dy = 1）。

## 精度与网格建议

- 默认 complex128（研究精度优先）；1024² 以上大网格可传
  `dtype`-风格的 float32 分量（VectorField 直接接受 complex64 数组）；
- 网格必须覆盖到 NA 光瞳映射后的焦斑范围（f·tanθ 与 f·sinθ 的差异
  在高 NA 下不可忽略）；
- 与标量框架相同：无带限滤波，wrap-around 风险自查（见
  [角谱传播](angular-spectrum.md) 的注意事项）。
