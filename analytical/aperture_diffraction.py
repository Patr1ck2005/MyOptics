import numpy as np
import matplotlib.pyplot as plt

# ---------------------- 参数定义 ----------------------

# 光学参数
wavelength = 1.550e-6*100  # 波长 (米)
k = 2 * np.pi / wavelength  # 波数 (rad/m)

# 几何参数
z1 = 0.01  # 点光源到圆孔的距离 (米)
z2 = 0.10  # 圆孔到观察平面的距离 (米)
# a = 25.4e-3  # 圆孔半径 (米)
a = 1e-3  # 圆孔半径 (米)

# 观察平面参数
N = 1024  # 观察平面网格点数
x_max = 0.01  # 观察平面横向范围 (米)
y_max = 0.01  # 观察平面纵向范围 (米)
x = np.linspace(-x_max, x_max, N)
y = np.linspace(-y_max, y_max, N)
X, Y = np.meshgrid(x, y)
R_obs = np.sqrt(X**2 + Y**2)  # 观察平面上的径向距离

# ---------------------- 衍射计算 ----------------------

# 预计算一些常量
const1 = k / (2 * z1)
const2 = k / (2 * z2)

# 圆孔离散化参数
Nr = 1024  # 圆孔径向采样点数
r = np.linspace(0, a, Nr)
dr = r[1] - r[0]

# 计算每个径向位置的贡献
# 使用数值积分时，考虑环形区域的面积
# 圆孔的每个环形区域的面积为 2πr dr

# 初始化复场
U = np.zeros((N, N), dtype=complex)

# 积分循环
for ri in r:
    # 计算相位延迟
    # 点光源到圆孔点的距离近似为 z1，因为 r << z1
    # 同理，观察点到圆孔点的距离近似为 z2
    # 完整路径长度 = z1 + z2 + (r^2)/(2z1) + ( (X - ri*np.cos(theta))^2 + (Y - ri*np.sin(theta))^2 ) / (2z2)
    # 为简化计算，假设小角度，径向相位近似：
    # 相位 = k (z1 + z2) + (k / (2 z1)) r^2 + (k / (2 z2)) R_obs^2
    # 由于 R_obs 依赖于 (X,Y)，需要更精确的计算

    # 计算环形区域对观察点的贡献
    # 采用轴对称假设，积分只需关于径向距离

    # 相位因子
    phi = k * (z1 + z2) + (k / (2 * z1)) * ri**2 + (k / (2 * z2)) * R_obs**2

    # 幅度贡献
    amplitude = ri * dr * 2 * np.pi  # 环形面积
    amplitude *= (1 / (1j * wavelength * z1)) * (1 / (1j * wavelength * z2))

    # 累加场
    U += amplitude * np.exp(1j * phi)

# # 考虑波源与观察平面距离
# U *= (1 / (1j * wavelength * z1)) * (1 / (1j * wavelength * z2))

# 计算光强
I = np.abs(U)**2

# 归一化光强
I /= np.max(I)

# ---------------------- 可视化 ----------------------

plt.figure(figsize=(8, 6))
extent = [-x_max*100, x_max*100, -y_max*100, y_max*100]  # 转换为厘米
plt.imshow(I, extent=extent, cmap='inferno')
plt.title('衍射光强分布 (观察平面距离圆孔 10 cm)')
plt.xlabel('x (cm)')
plt.ylabel('y (cm)')
plt.colorbar(label='归一化光强')
plt.show()
