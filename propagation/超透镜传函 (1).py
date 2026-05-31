# -*- coding: gbk -*-
import numpy as np
import numpy.lib.scimath as smath
import cmath

def calculate_H(k_x, lamda, z):
    """
    输入参数：
    k_x  : 波矢x分量 (1/m)
    wavelength: 波长 (m)
    z    : 位置坐标数组 (numpy array, 单位m)
    
    输出：
    H    : 场分布数组 (与z同形)
    """
    # 初始化参数（后续可修改epsilon）
    epsilon = np.array([2.56,-2.6115+0.4431*1j,2.7640+0.1808*1j,-2.6194+0.4551*1j,2.43])  # 示例值，可修改
    
    # 计算kz和K ------------------------------------------------------------
    kz = np.array([
        smath.sqrt((epsilon[0])*(2*np.pi/lamda)**2 - k_x**2),
        smath.sqrt((epsilon[1])*(2*np.pi/lamda)**2 - k_x**2),
        smath.sqrt((epsilon[2])*(2*np.pi/lamda)**2 - k_x**2),
        smath.sqrt((epsilon[3])*(2*np.pi/lamda)**2 - k_x**2),
        smath.sqrt((epsilon[4])*(2*np.pi/lamda)**2 - k_x**2)
    ])
    K = kz / epsilon
    
    # 层参数计算函数 -------------------------------------------------------
    def calculate_layer(l, d, epsilon, K, kz):
        """计算单层传输参数"""
        # 索引转换 (MATLAB 1-based -> Python 0-based)
        eps_l3 = epsilon[l+3-1]  # 原MATLAB epsilon(l+3)
        eps_l1 = epsilon[l+1-1]  # 原MATLAB epsilon(l+1)
        K_l2 = K[l+2-1]         # 原MATLAB K(l+2)
        K_l3 = K[l+3-1]         # 原MATLAB K(l+3)
        K_l1 = K[l+1-1]         # 原MATLAB K(l+1)
        kz_l2 = kz[l+2-1]       # 原MATLAB kz(l+2)

        # 计算传输矩阵
        numerator = 4*K_l2*K_l3 * cmath.exp(1j*d*kz_l2)
        denominator = (cmath.exp(2j*d*kz_l2)*(K_l1 - K_l2)*(K_l2 - K_l3) 
                      + (K_l1 + K_l2)*(K_l2 + K_l3))
        T = eps_l3/eps_l1 * numerator / denominator
        
        # 计算反射系数
        R_num = ( (K_l1 + K_l2)*(K_l2 - K_l3)*cmath.exp(2j*kz_l2*d)
                + (K_l1 - K_l2)*(K_l2 + K_l3) )
        R_den = ( (K_l1 - K_l2)*(K_l2 - K_l3)*cmath.exp(2j*kz_l2*d)
                + (K_l1 + K_l2)*(K_l2 + K_l3) )
        R = R_num / R_den
        
        return T, R

    # 计算各层参数 ---------------------------------------------------------
    T_1, R_1 = calculate_layer(l=0, d=1e-8, epsilon=epsilon, K=K, kz=kz)
    T_2, R_2 = calculate_layer(l=2, d=4e-8, epsilon=epsilon, K=K, kz=kz)
    
    l=0
    d=1e-8
    eps_l3 = epsilon[l+3-1]  # 原MATLAB epsilon(l+3)
    eps_l1 = epsilon[l+1-1]  # 原MATLAB epsilon(l+1)
    K_l2 = K[l+2-1]         # 原MATLAB K(l+2)
    K_l3 = K[l+3-1]         # 原MATLAB K(l+3)
    K_l1 = K[l+1-1]         # 原MATLAB K(l+1)
    kz_l2 = kz[l+2-1]       # 原MATLAB kz(l+2)
    numerator = 4*K_l1*K_l2 * cmath.exp(1j*d*kz_l2)
    denominator = (cmath.exp(2j*d*kz_l2)*(K_l1 - K_l2)*(K_l2 - K_l3) 
                      + (K_l1 + K_l2)*(K_l2 + K_l3))
    T_3 = eps_l1/eps_l3 * numerator / denominator
        
        # 计算反射系数
    R_num = ( (K_l1 - K_l2)*(K_l2 + K_l3)*cmath.exp(2j*kz_l2*d)
                + (K_l1 +K_l2)*(K_l2 - K_l3) )
    R_den = ( (K_l1 - K_l2)*(K_l2 - K_l3)*cmath.exp(2j*kz_l2*d)
                + (K_l1 + K_l2)*(K_l2 + K_l3) )
    R_3 = -R_num / R_den
    
    
    
    
    # 总体传输和反射系数 ----------------------------------------------------
    l = 1
    d_layer = 1.5e-8
    kz_l2 = kz[l+2-1]
    
    denominator = 1 - R_2*R_3*cmath.exp(2j*kz_l2*d_layer)
    T = (T_1*T_2*cmath.exp(1j*kz_l2*d_layer)) / denominator
    R = R_1 + (T_1*R_2*T_3*cmath.exp(2j*kz_l2*d_layer)) / denominator
    
    T_4 = T / (T_2*cmath.exp(1j*kz_l2*d_layer))
    R_4 = (R - R_1) / (T_3*cmath.exp(1j*kz_l2*d_layer))

    # 求解系数矩阵 ---------------------------------------------------------
    def solve_coeff(kz_layer, z0, d, rhs1, rhs2):
        """解线性方程组 AX = B """
        a = kz_layer
        coeff_matrix = np.array([
            [cmath.exp(1j*a*z0), cmath.exp(-1j*a*z0)],
            [cmath.exp(1j*a*(z0 + d)), cmath.exp(-1j*a*(z0 + d))]
        ], dtype=complex)
        return np.linalg.solve(coeff_matrix, [rhs1, rhs2])

    # 计算A,B
    z0_AB = 1e-8
    d_AB = 1e-8
    kz_AB = kz[1]  # kz(2)
    rhs1_AB = cmath.exp(1j*kz[0]*(z0_AB - 1e-8)) + R*cmath.exp(-1j*kz[0]*(z0_AB - 1e-8))
    rhs2_AB = T_4*cmath.exp(1j*kz[2]*(z0_AB + d_AB - 2e-8)) + \
              R_4*cmath.exp(-1j*kz[2]*(z0_AB + d_AB - 3.5e-8))
    A, B = solve_coeff(kz_AB, z0_AB, d_AB, rhs1_AB, rhs2_AB)

    # 计算C,D
    z0_CD = 3.5e-8
    d_CD = 4e-8
    kz_CD = kz[3]  # kz(4)
    rhs1_CD = T_4*cmath.exp(1j*kz[2]*(z0_CD - 2e-8)) + \
              R_4*cmath.exp(-1j*kz[2]*(z0_CD - 3.5e-8))
    rhs2_CD = T*cmath.exp(1j*kz[4]*(z0_CD + d_CD - 7.5e-8))
    C, D = solve_coeff(kz_CD, z0_CD, d_CD, rhs1_CD, rhs2_CD)

    # 根据z的位置计算H -----------------------------------------------------
    H = np.zeros_like(z, dtype=complex)
    K_1 = 1.0  # 原K_1系数
    
    # 定义各区间条件
    cond1 = z < 1e-8                      # 0-1e-8
    cond2 = (1e-8 <= z) & (z < 2e-8)      # 1e-8-2e-8
    cond3 = (2e-8 <= z) & (z < 3.5e-8)    # 2e-8-3.5e-8
    cond4 = (3.5e-8 <= z) & (z < 7.5e-8)  # 3.5e-8-7.5e-8
    cond5 = z >= 7.5e-8                   # 7.5e-8-9e-8
    
    # 各区间计算公式
    H[cond1] = K_1*(np.exp(1j*kz[0]*(z[cond1] - 1e-8)) + R*np.exp(-1j*kz[0]*(z[cond1] - 1e-8)))
    H[cond2] = K_1*(A*np.exp(1j*kz[1]*z[cond2]) + B*np.exp(-1j*kz[1]*z[cond2]))
    H[cond3] = K_1*(T_4*np.exp(1j*kz[2]*(z[cond3] - 2e-8)) + R_4*np.exp(-1j*kz[2]*(z[cond3] - 3.5e-8)))
    H[cond4] = K_1*(C*np.exp(1j*kz[3]*z[cond4]) + D*np.exp(-1j*kz[3]*z[cond4]))
    H[cond5] = K_1*T*np.exp(1j*kz[4]*(z[cond5] - 7.5e-8))
    
    return np.abs(H)  # 返回场强的模值

# 使用示例 ----------------------------------------------------------------
if __name__ == "__main__":
    # 生成自定义z数组（示例：0到9e-8之间，步长1e-10）
    z_external = np.arange(0, 9e-8, 1e-10)
    
    # 计算场分布
    H_result = calculate_H(
        k_x=5.3337e7,
        lamda=365e-9, 
        z=z_external
    )
    
    # 绘图
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(z_external, H_result)
    plt.xlabel('Position (m)')
    plt.ylabel('Field Magnitude')
    plt.title('Custom Z Input Field Distribution')
    plt.show()