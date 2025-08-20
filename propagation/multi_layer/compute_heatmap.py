from cls import *

import numpy as np
import matplotlib.pyplot as plt

# 假设你的 Layer 和 MultiLayerTM 已经定义在上面……

if __name__ == '__main__':
    # 参数设置
    z0 = 75.0

    # 第三层厚度范围（nm）
    die_thicks = np.linspace(0, 40, 100)
    # k0 范围：这里以原始 k0 为中心 ±50%
    k0_center = 2*np.pi / 365
    k0_vals = np.linspace(-10*k0_center, 10*k0_center, 200)

    # 结果矩阵：行对应 die_thicks，列对应 k0_vals
    intensity_map = np.zeros((die_thicks.size, k0_vals.size))

    # 扫描双参数
    for i, d in enumerate(die_thicks):
        for j, k0 in enumerate(k0_vals):
            # 使用 k0 构造对应的波长
            layers = [
                Layer(2.56, 10),
                Layer(-2.6115 + 0.4431j, 40),
                Layer(2.7640 + 0.1808j, d),
                Layer(-2.6194 + 0.4551j, 40),
                Layer(2.43, None),
            ]
            solver = MultiLayerTM(layers,
                                  eps_incident=2.56,
                                  wavelength=365,
                                  kx=k0)
            # 直接拿强度
            intensity_map[i, j] = solver.field_intensity_at(z0)

    # 绘制 2D 颜色映射
    X, Y = np.meshgrid(k0_vals, die_thicks)
    plt.figure(figsize=(8,6))
    pcm = plt.pcolormesh(X, Y, intensity_map,
                         shading='auto', cmap='hot', vmin=0, vmax=10)
    plt.colorbar(pcm, label=r'$|H(z=75)|^2$')
    plt.xlabel(r'$k_0$ (rad/nm)')
    plt.ylabel('die thickness (nm)')
    plt.title(r'Field Intensity at $z=75$ vs. die\_thickness & $k_0$')
    plt.tight_layout()
    plt.show()

