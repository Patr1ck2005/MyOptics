import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift, ifftshift, ifft2
from scipy.ndimage import gaussian_filter


def generate_reference_wave(shape, intensity, curvature_radius, wavelength, pixel_size):
    """
    生成参考球面波的相位。

    参数:
    - shape: 图像的形状 (高度, 宽度)
    - curvature_radius: 球面波的曲率半径 (单位与pixel_size相同)
    - wavelength: 光波长
    - pixel_size: 每个像素的物理尺寸
    """
    ny, nx = shape
    y, x = np.indices((ny, nx))
    center_y, center_x = ny // 2, nx // 2
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) * pixel_size
    ref_wave = intensity * np.exp(1j * 2 * np.pi * r ** 2 / (2 * curvature_radius * wavelength))
    return ref_wave


# Step 1: 加载干涉图样
interference_intensity_noisy = np.load('./interference_pattern.npy')
N = interference_intensity_noisy.shape[0]

# 显示原始干涉图像
plt.figure(figsize=(6, 6))
plt.imshow(interference_intensity_noisy, cmap='gray', vmin=0)
plt.title('Original Interference Pattern')
plt.axis('off')
plt.show()

# Step 2: 预处理图像 - 傅里叶滤波以去除不需要的干涉光束
# 进行傅里叶变换
F = fftshift(fft2(interference_intensity_noisy))

# 显示傅里叶谱
plt.figure(figsize=(6, 6))
# plt.imshow(np.log(np.abs(F) + 1), cmap='gray')
plt.imshow(np.angle(F), cmap='twilight')
plt.colorbar()
plt.title('Fourier Spectrum of Interference Pattern')
plt.axis('off')
plt.show()

# 根据傅里叶谱手动选择合适的滤波窗口
# 这里假设主干涉条纹位于中心，可以调整掩码以适应实际情况
ny, nx = interference_intensity_noisy.shape
mask = np.zeros((ny, nx))
# 定义一个圆形掩码，半径根据实际情况调整
radius = 10
y, x = np.ogrid[:ny, :nx]
center_y, center_x = ny // 2, nx // 2

loc_x = 546
loc_y = 512
mask_area = (y - loc_y) ** 2 + (x - loc_x) ** 2 <= radius ** 2
mask[mask_area] = 1

# 应用掩码
F_filtered = F * mask
# 平移傅里叶谱
F_filtered = np.roll(F_filtered, shift=(center_y-loc_y, center_x-loc_x))
# # 显示傅里叶谱
# plt.figure(figsize=(6, 6))
# # plt.imshow(np.log(np.abs(F_filtered) + 1), cmap='gray')
# plt.imshow(np.angle(F_filtered), cmap='twilight')
# plt.colorbar()
# plt.title('Fourier Spectrum of Interference Pattern')
# plt.axis('off')
# plt.show()

# 逆傅里叶变换回空间域
interference_filtered = ifft2(ifftshift(F_filtered))
interference_filtered_phase = np.angle(interference_filtered)

# 显示最终提取的待测光束
plt.subplots(1, 2, figsize=(12, 6))
plt.subplot(121)
plt.imshow(np.abs(interference_filtered)**2, cmap='gray', vmin=0)
plt.title('Extracted Intensity')
plt.colorbar()
plt.subplot(122)
plt.imshow(np.angle(interference_filtered), cmap='twilight')
plt.title('Extracted Phase')
plt.colorbar()
plt.tight_layout()
plt.show()

