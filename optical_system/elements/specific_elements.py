import numpy as np
import cupy as cp

from optical_system.elements import MomentumSpacePlate, MomentumSpaceModulator
from utils.constants import PI

import os

# 获取当前文件 (element.py) 所在目录的路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构建 data 文件夹中 efficiency.npy 的绝对路径
data_path = lambda file_name: os.path.join(current_dir, "data", file_name)

class SimpleMSPP(MomentumSpacePlate):
    def __init__(self, z_position, topology_charge, wavelength, inner_NA=0, outer_NA=None):
        """
        初始化动量空间的相位板。

        参数:
        z_position (float): 相位板在z轴上的位置。
        topology_charge (int): 涡旋的拓扑荷数。
        wavelength (float): 光波波长。
        inner_NA (float): 环形转换区域的内数值孔径 (NA)。
        outer_NA (float): 环形转换区域的外数值孔径 (NA)。如果为 None，则设置为最大可能值。
        """
        # 定义相位函数
        def phase_function(KX, KY):
            # 计算动量空间径向分量
            K_magnitude = cp.sqrt(KX**2 + KY**2)

            # 计算波矢量大小 (k = 2π / λ)
            k = 2 * PI / wavelength

            # 将NA转换为动量空间的径向范围
            inner_k = inner_NA * k
            outer_k = outer_NA * k if outer_NA is not None else cp.inf

            # 构造环形转换区域的掩模
            mask = (K_magnitude >= inner_k) & (K_magnitude <= outer_k)

            # 计算涡旋相位因子 (基于拓扑荷数)
            vortex_phase = cp.exp(1j * topology_charge * cp.arctan2(KY, KX))

            # 将掩模应用到涡旋相位
            return cp.where(mask, vortex_phase, 0)

        # 初始化父类
        super().__init__(z_position, phase_function)


class MSPP(MomentumSpaceModulator):
    def __init__(self, z_position, wavelength):
        # 加载动量空间调制器数据
        # efficiency = np.load("./data/efficiency.npy")  # 加载效率数组
        # phase = np.load("./data/phase.npy")  # 加载相位数组
        efficiency = np.load(data_path('efficiency.npy'))  # 加载效率数组
        phase = np.load(data_path('phase.npy'))  # 加载相位数组

        # 计算总调制因子：sqrt(efficiency) * exp(i * phase)
        modulation_array = np.sqrt(efficiency) * np.exp(1j * phase)

        # 数据来源的动量空间坐标网格定义
        data_NA = np.sin(np.deg2rad(40))  # 数值孔径
        data_k_max = 2 * PI * data_NA / wavelength
        n_k = efficiency.shape[0]
        data_kx = np.linspace(-data_k_max, data_k_max, n_k)
        data_ky = np.linspace(-data_k_max, data_k_max, n_k)

        # 初始化光调制器
        super().__init__(z_position, modulation_array=modulation_array, mod_kx=data_kx, mod_ky=data_ky)