"""VectorOpticalSystem：矢量光场端到端仿真（光源 → 元件序列 → 传播）。

与标量 OpticalSystem 相同的编排模式：元件按 z_position 排序，传播到
元件位置时应用元件，最后到达各目标 z。区别：
- 场是三分量 VectorField；
- 传播器是 VectorAngularSpectrumPropagator（每步横场投影 + Ez 重构）；
- Jones 薄元件作用后 ez=None，由下一次传播自动重构（包括 z 落点处的
  零距离 project——保证任何输出平面都带完整三分量）。

功率约定与标量框架一致：初始化时总功率归一到 1（Σ|E|²·dx·dy = 1）。
"""
import logging

import cupy as cp
import numpy as np
from tqdm import tqdm

from vector.elements import VectorElement
from vector.field import VectorField
from vector.propagator import VectorAngularSpectrumPropagator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def _to_numpy_field(field: VectorField) -> VectorField:
    """把 VectorField 的数组转换为 numpy（结果离场）。"""
    return VectorField(
        cp.asnumpy(field.ex), cp.asnumpy(field.ey),
        None if field.ez is None else cp.asnumpy(field.ez),
        cp.asnumpy(field.x), cp.asnumpy(field.y), field.wavelength,
    )


class VectorOpticalSystem:
    """矢量光学系统。"""

    def __init__(self, wavelength, x, y, initial_field: VectorField,
                 normalize=True):
        """
        参数:
        wavelength (float): 波长。
        x, y (ndarray): 1D 坐标。
        initial_field (VectorField): 初始矢量场（横向分量必须给出）。
        normalize (bool): 总功率归一到 1。
        """
        self.wavelength = float(wavelength)
        self.x = cp.asarray(x)
        self.y = cp.asarray(y)
        field = initial_field.copy()
        field.x, field.y = self.x, self.y
        self.E = field.normalized() if normalize else field
        self.elements: list[VectorElement] = []
        self.element_positions: list[float] = []
        self.sorted = False
        logging.info("VectorOpticalSystem initialized: power=%.6f", self.E.power())

    # ------------------------------------------------------------------
    def add_element(self, element: VectorElement):
        self.elements.append(element)
        self.element_positions.append(element.z_position)
        self.sorted = False
        logging.info("Added vector element at z=%.2f", element.z_position)

    def sort_elements(self):
        if not self.sorted:
            pairs = sorted(zip(self.element_positions, self.elements),
                           key=lambda pair: pair[0])
            if pairs:
                self.element_positions, self.elements = (list(p) for p in zip(*pairs))
            else:
                self.element_positions, self.elements = [], []
            self.sorted = True

    # ------------------------------------------------------------------
    def propagate_to_cross_sections(self, z_positions):
        """传播到指定 z 平面，返回 {z: VectorField(numpy)}（含完整三分量）。"""
        self.sort_elements()
        prop = VectorAngularSpectrumPropagator(self.x, self.y, self.wavelength)
        results = {}
        current = self.E.copy()
        current_z = 0.0
        idx = 0

        for z in tqdm(sorted(np.asarray(z_positions).tolist()),
                      desc="Vector propagating to cross sections"):
            while idx < len(self.elements) and self.element_positions[idx] <= z:
                z_elem = self.element_positions[idx]
                if z_elem > current_z:
                    current = prop.propagate(current, z_elem - current_z)
                    current_z = z_elem
                current = self.elements[idx].apply(current)
                idx += 1
            if z > current_z:
                current = prop.propagate(current, z - current_z)
                current_z = z
            else:
                current = prop.project(current)   # 元件后零距离重构 Ez
            results[float(z)] = _to_numpy_field(current)
            current = current.copy()
        return results

    def propagate_to_longitudinal(self, direction='x', position=0.0,
                                  num_z=200, z_max=100.0):
        """纵向强度截面（不落盘整场，显存安全）。

        返回 (coord_axis, z_coords, I_total(n_line, nz),
              I_components: {'ex':..., 'ey':..., 'ez':...})，均为 numpy。
        """
        if direction not in ('x', 'y'):
            raise ValueError("direction 必须是 'x' 或 'y'")
        self.sort_elements()
        prop = VectorAngularSpectrumPropagator(self.x, self.y, self.wavelength)
        z_coords = np.linspace(0.0, z_max, num_z)
        axis_arr = self.x if direction == 'x' else self.y
        coord_axis = np.asarray(cp.asnumpy(axis_arr))
        pos_idx = int(cp.argmin(cp.abs(axis_arr - position)).get())

        n_line = self.E.nx if direction == 'x' else self.E.ny
        I_total = np.zeros((n_line, num_z))
        I_comp = {k: np.zeros((n_line, num_z)) for k in ('ex', 'ey', 'ez')}

        current = self.E.copy()
        current_z = 0.0
        idx = 0
        for i, z in enumerate(tqdm(z_coords, desc="Vector longitudinal section")):
            while idx < len(self.elements) and self.element_positions[idx] <= z:
                z_elem = self.element_positions[idx]
                if z_elem > current_z:
                    current = prop.propagate(current, z_elem - current_z)
                    current_z = z_elem
                current = self.elements[idx].apply(current)
                idx += 1
            if z > current_z:
                current = prop.propagate(current, z - current_z)
                current_z = z
            line = _line_slice(current, direction, pos_idx)
            I_total[:, i] = line['total']
            for k in I_comp:
                I_comp[k][:, i] = line[k]
        return coord_axis, z_coords, I_total, I_comp


def _line_slice(field: VectorField, direction: str, pos_idx: int):
    """取纵截面线上各分量强度（返回 numpy dict）。"""
    if direction == 'x':
        ex, ey = field.ex[:, pos_idx], field.ey[:, pos_idx]
        ez = field.ez[:, pos_idx] if field.ez is not None else None
    else:
        ex, ey = field.ex[pos_idx, :], field.ey[pos_idx, :]
        ez = field.ez[pos_idx, :] if field.ez is not None else None

    def _i(u):
        if u is None:
            return None
        arr = cp.abs(u) ** 2 if isinstance(u, cp.ndarray) else np.abs(u) ** 2
        return cp.asnumpy(arr) if isinstance(arr, cp.ndarray) else arr

    iex, iey = _i(ex), _i(ey)
    iez = _i(ez) if ez is not None else np.zeros_like(iex)
    return {'ex': iex, 'ey': iey, 'ez': iez, 'total': iex + iey + iez}
