
import numpy as np
import matplotlib.pyplot as plt

# =============================
# 参数设置
# =============================
fs = 9
plt.rcParams['font.size'] = fs

# Chinese fonts setup (uncomment if needed)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 负号显示
# 掩模占空比：例如 0.3, 0.5, 0.7
fill_factors = [0.3, 0.5, 0.7]

# x 轴：以周期 Λ 归一化，这里设 Λ = 1
Lambda = 1.0
x = np.linspace(-1.5 * Lambda, 1.5 * Lambda, 2000)  # 跨越多个周期

# 计算衍射级次的最大阶数（越大越接近真实，示意图取 10 足够）
N_orders = 10


# =============================
# 计算一维周期性掩模的像场剂量分布
# 使用简单的傅里叶光栅模型：
# 掩模开口为矩形函数，开口占空比为 f
# 其傅里叶系数 C_n = f * sinc(n * f) * exp(-i * pi * n * f)
# 像场（标量近轴）取为各衍射级的叠加：
# E(x) = sum_n C_n * exp(i * 2π n x / Λ)
# 像场强度 I(x) ∝ |E(x)|^2，此处归一化为最大值 1
# =============================

def aerial_image_1d(x, f, Lambda=1.0, N_orders=10):
    """
    计算给定占空比 f 下的一维周期性掩模像场强度（归一化）。
    x: 空间坐标
    f: 占空比 (0~1)
    Lambda: 周期，设为 1 即表示 x 以 Λ 归一化
    N_orders: 衍射级次数目，正负各 N_orders 阶
    """
    # 初始化复振幅
    E = np.zeros_like(x, dtype=complex)

    # n = -N_orders ... N_orders
    for n in range(-N_orders, N_orders + 1):
        # 矩形光栅的傅里叶系数（使用 numpy 的 sinc(x)=sin(pi x)/(pi x)）
        C_n = f * np.sinc(n * f) * np.exp(-1j * np.pi * n * f)
        E += C_n * np.exp(1j * 2 * np.pi * n * x / Lambda)

    I = np.abs(E) ** 2
    # 归一化到最大值 1
    I /= I.max()
    return I


# =============================
# 绘图
# =============================
plt.figure(figsize=(4, 2), dpi=150)

for f in fill_factors:
    I = aerial_image_1d(x, f, Lambda=Lambda, N_orders=N_orders)
    # 剂量 D(x) ∝ I(x)，这里直接用归一化的 I(x) 作为 D(x)/D0
    plt.plot(x / Lambda, I, linewidth=1.5, label=fr'$f = {f}$')

plt.xlabel(r'$x / \Lambda$')
plt.ylabel(r'归一化剂量 $D(x)/D_0$')
plt.title('不同掩模占空比下像场剂量分布示意图')

plt.xlim(-1.5, 1.5)
plt.ylim(0, 1.05)
plt.grid(True, linestyle=':', linewidth=0.5)
plt.legend(loc='upper right', fontsize=9)
plt.tight_layout()

# 保存图像到论文中预留的位置
plt.savefig('dose-vs-mask.png', dpi=300, bbox_inches='tight')
plt.show()
