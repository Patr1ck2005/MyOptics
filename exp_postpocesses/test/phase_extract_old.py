import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration.unwrap import unwrap_phase


# 假设 generate_reference_wave 函数已定义
def generate_reference_wave(shape, amplitude_ref, curvature_radius, wavelength, pixel_size):
    """
    生成参考波的复振幅。
    """
    y, x = np.indices(shape)
    center = (shape[0] // 2, shape[1] // 2)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    phase = (2 * np.pi / wavelength) * r**2 / (2 * curvature_radius)
    return amplitude_ref * np.exp(1j * phase)

# 加载已处理的干涉图像
# interference = np.load('./interference_filtered.npy')
interference = np.load('./interference_pattern.npy')
plt.imshow(interference, vmin=0)
plt.title('interference')
plt.colorbar()
plt.show()

# 已知参数
N = interference.shape[0]  # 假设图像是方形的
curvature_radius = 1 * N  # 参考波曲率半径，单位：像素
wavelength = 0.01 * N       # 波长，单位：像素
amplitude_ref = 1         # 参考波的振幅
pixel_size = 1.0            # 每个像素的物理尺寸

# 已知参考波复振幅
ref_wave = generate_reference_wave(
    interference.shape,
    amplitude_ref,
    curvature_radius,
    wavelength,
    pixel_size
)

# 已知待测波干涉前强度 |A|^2
before_interference = np.load('before_interference.npy')  # |A|^2

# 计算参考波的强度 |R|^2
R_magnitude = np.abs(ref_wave)
R_phase = np.angle(ref_wave)
R_intensity = R_magnitude**2

# 计算待测波的幅度 |A|
A_intensity = before_interference  # |A|^2
A_magnitude = np.sqrt(A_intensity)

# 计算干涉项 Re(A R*)
# I = |A|^2 + |R|^2 + 2 Re(A R*)
Re_A_R = (interference - A_intensity - R_intensity) / 2

# 计算 cos(phi_A - phi_R)
cos_phi_diff = Re_A_R / (A_magnitude * R_magnitude)

# 为避免数值问题，将 cos_phi_diff 限制在 [-1, 1]
cos_phi_diff_clipped = np.clip(cos_phi_diff, -1.0, 1.0)
plt.imshow(cos_phi_diff)
plt.title('cos_phi_diff')
plt.colorbar()
plt.show()

# 计算相位差的符号，根据已知拓扑电荷=2来扩展相位差到 [0, 2pi]

# 获取图像中心
center_y, center_x = N // 2, N // 2

# 创建坐标网格
y, x = np.indices(interference.shape)
delta_x = x - center_x
delta_y = y - center_y

# 计算极坐标角度 theta，范围 [0, 2pi)
theta = np.arctan2(delta_y, delta_x)
theta = np.mod(theta, 2 * np.pi)

# 计算期望的相位差 phi_diff_expected = l * theta
l = 2  # 拓扑电荷
phi_diff_expected = l * theta  # 拓扑电荷=2
#
# # 计算相位差 phi_A - phi_R
# if phi_diff < phi_diff_expected:
phi_diff_expected_mask = np.sign(np.cos(phi_diff_expected+np.pi/2))

phi_diff = np.arccos(cos_phi_diff_clipped)  # 取值范围 [0, pi]
# phi_diff
phi_diff %= 2*np.pi
plt.imshow(phi_diff, cmap='twilight', vmin=0, vmax=2*np.pi)
# plt.imshow(phi_diff, cmap='twilight', vmin=0, vmax=1*np.pi)
# plt.imshow(phi_diff, cmap='twilight')
plt.title('phi_diff')
plt.colorbar()
plt.show()


# 计算待测波的相位 phi_A = phi_diff + phi_R
phi_A = phi_diff + R_phase
phi_A %= 2*np.pi

# 可视化相位分布
plt.figure(figsize=(8,6))
plt.imshow(phi_A, cmap='twilight')
plt.colorbar(label='Phase (radians)')
plt.title('Vortex Phase Distribution')
plt.xlabel('X Pixel')
plt.ylabel('Y Pixel')
plt.show()

