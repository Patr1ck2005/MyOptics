# 角谱传播（Angular Spectrum Method）

## 原理

角谱法把平面 $z=0$ 上的复场 $U(x,y)$ 分解为平面波谱：

$$\tilde U(k_x, k_y) = \iint U(x,y)\, e^{-i(k_x x + k_y y)}\, dx\, dy$$

每个平面波分量沿 $z$ 传播时获得相位 $e^{i k_z z}$，其中

$$k_z = \sqrt{k_0^2 - k_x^2 - k_y^2}, \qquad k_0 = \frac{2\pi}{\lambda}$$

传播到任意平面就是一次频域滤波：

$$U(x,y;z) = \mathcal{F}^{-1}\{\, e^{i k_z z}\, \mathcal{F}[U]\,\}$$

- **传播带**（$k_x^2+k_y^2 < k_0^2$）：$k_z$ 为实数，纯相位传播，能量守恒。
- **倏逝带**（$k_x^2+k_y^2 > k_0^2$）：$k_z = i\gamma$，振幅按 $e^{-\gamma z}$
  衰减——亚波长细节信息在这部分，衰减快但近场完整。

`Fresnel` 模式用近轴近似 $k_z \approx k_0 - \frac{\pi\lambda}{k_0'}(f_x^2+f_y^2)$，
仅适用于小角度；研究默认用 `Rigorous`。

## 采样定理与窗口选择

| 参数 | 关系 | 后果 |
|------|------|------|
| 网格间距 $\Delta x$ | 决定最高空间频率 $f_{max} = 1/(2\Delta x)$ | 太疏 → 高频混叠，倏逝细节丢失 |
| 窗口 $L$ | 决定频率分辨率 $\Delta f = 1/L$ | 太小 → 光束撞墙，wrap-around 混叠 |
| 网格数 $N$ | $N = L/\Delta x$ 同时约束两者 | 大 $N$ = 显存/时间代价 |

!!! warning "wrap-around"
    角谱传播隐含周期边界。长距离传播时务必让窗口显著大于光束路径，
    否则出窗的场会从对侧卷回。框架**不做**自动带限或吸收边界。

## 缓存传播器

逐次调用 `angular_spectrum_propagate` 每步都重建 fftfreq/meshgrid/kz 网格。
`AngularSpectrumPropagator` 在构造时缓存：

```python
from propagation.propagator import AngularSpectrumPropagator

prop = AngularSpectrumPropagator(x, y, wavelength, mode='Rigorous')
U1 = prop.propagate(U, z=10.0)
series = prop.propagate_batch(U, z_list=[5, 10, 15, 20])   # 复用同一份缓存
```

- `propagate(U, z)` 每步只剩一次 `exp(1j·kz·z)` + FFT 对；
- `prop.X / prop.Y` 实空间网格可复用给元件调制；
- 纵向多步扫描（`OpticalSystem.propagate_to_longitudinal_section` 内部
  已接入）实测纯传播 **2.68×** 提速（513² × 256 步，RTX 4060）；
- 与旧实现**逐位一致**（回归测试 `tests/test_propagator.py` 保证）。

## 精度

默认 `complex128`。float32 下大光程相位积累的舍入在焦场和干涉结构上
肉眼可见——研究计算不要用；速度敏感的批量扫描可显式降级。
