import numpy as np
import matplotlib.pyplot as plt

fs = 9
plt.rcParams['font.size'] = fs

# Chinese fonts setup (uncomment if needed)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 负号显示

# 1. 定义曝光剂量范围（单位随意，比如 mJ/cm^2）
E = np.linspace(0, 200, 500)   # 从 0 到 200

# 2. 定义 Sigmoid 型响应（这里以正性光刻胶为例：剂量越大，剩余厚度越小）
E0 = 80      # 阈值剂量/转折点
k  = 0.08    # 曲线陡峭程度

# Sigmoid 函数：剩余厚度归一化 T/T0
T_norm = 1 / (1 + np.exp(k * (E - E0)))  # 递减型 Sigmoid

# 3. 画图
plt.figure(figsize=(3, 2), dpi=120)

plt.plot(E, T_norm)

# 标出阈值剂量位置
plt.axvline(E0, linestyle='--')
plt.text(E0 + 2, 0.55, r'$E_0$')

# 坐标轴与标签
plt.xlabel('曝光剂量  E (a.u.)')
plt.ylabel('光刻胶剩余厚度  T/T0')
plt.title('Sigmoid 型光刻胶响应曲线示意')

plt.xlim(0, 200)
plt.ylim(-0.05, 1.05)

plt.grid(True, linestyle=':')
# plt.tight_layout()
plt.savefig('sigmoid_photoresist_response.png', dpi=300, bbox_inches='tight')
plt.show()
