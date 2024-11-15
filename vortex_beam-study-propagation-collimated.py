# main.py

import numpy as np
from optical_system.system import OpticalSystem
from optical_system.elements import Lens, PhasePlate, MomentumSpacePhasePlate
from visualization.plotter import Plotter
from utils.constants import PI

def main():
    # 定义仿真参数
    wavelength = 0.8  # 单位：微米
    theta = np.deg2rad(0.68)
    w_0 = wavelength/theta/PI
    z_0 = PI*w_0**2/wavelength
    d = 1*z_0
    sim_size = 50*d*np.tan(theta)
    mesh = 1024*6+1
    x = np.linspace(-sim_size, sim_size, mesh)
    y = np.linspace(-sim_size, sim_size, mesh)
    X, Y = np.meshgrid(x, y)

    # 定义初始光场（高斯光束）
    initial_field = np.exp(- (X**2 + Y**2) / w_0**2)

    # 创建光学系统
    optical_system = OpticalSystem(wavelength, x, y, initial_field)

    # # 添加光学元件
    optical_system.add_element(MomentumSpacePhasePlate(z_position=0, phase_function=lambda KX, KY: np.exp(1j * 1/2 * np.arctan2(KY, KX))))
    # optical_system.add_element(PhasePlate(z_position=0, phase_function=lambda X, Y: np.exp(1j * 1/2 * np.arctan2(Y, X))))
    # optical_system.add_element(Lens(z_position=f, focal_length=f/2))

    save_label = 'mystructer-collimate-Rigorous'
    # save_label = 'SSP-collimated-Rigorous'

    # 创建绘图器
    plotter = Plotter(x, y)

    # 计算并绘制横截面
    cross_z_positions = [0, 0.001*z_0, 0.5*z_0, d]  # 需要计算的z位置
    cross_sections = optical_system.propagate_to_cross_sections(cross_z_positions,
                                                                propagation_mode='Rigorous',
                                                                return_momentum_space_spectrum=True)
    plotter.plot_cross_sections(cross_sections,
                                save_label=save_label,
                                show=False)

    # 计算并绘制纵截面
    # 例如，沿x方向，在x=0的位置
    direction = 'x'
    position = 0.0
    num_z = 128*2
    z_max = d
    coord_axis, z_coords, intensity, phase = optical_system.propagate_to_longitudinal_section(
        direction=direction,
        position=position,
        num_z=num_z,
        z_max=z_max,
        propagation_mode='Rigorous',
    )
    plotter.plot_longitudinal_section(coord_axis, z_coords, intensity, phase,
                                      direction=direction,
                                      position=position,
                                      save_label=save_label,
                                      show=False)


if __name__ == "__main__":
    main()
