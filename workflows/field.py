"""C4 Field：仿真结果的标准容器（元数据 + 强度/复场 + 持久化）。

设计动机：框架各仿真入口（OpticalSystem / VectorOpticalSystem /
RichardsWolfFocuser）的返回格式互不相同（tuple / dict / dataclass），
下游的参数扫描与宽谱合成需要一个统一的结果类型。Field 只持有
**numpy 数组**（离场快照），是数据边界——GPU 张量不进入本容器。

- ``intensities``: 命名强度分量（标量仿真 {'U'}；矢量仿真
  {'ex','ey','ez'}）；``total_intensity`` 为各分量之和；
- ``complex_fields``: 可选的复场快照（宽谱相干合成必需；
  大扫描建议丢弃以省内存）；
- ``save/load``: 单文件 npz（压缩），元数据 JSON 编码；
- ``line_cut``: 过峰值/指定位置的截线（绘图与 FWHM 的底层数据）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _normalize_z(system, z_positions):
    """z_positions 归一：None → 系统最后元件位置；标量 → 单元素列表。"""
    if z_positions is None:
        system.sort_elements()
        return [system.element_positions[-1]] \
            if system.element_positions else [0.0]
    if np.isscalar(z_positions):
        return [float(z_positions)]
    return list(z_positions)


@dataclass
class Field:
    """一次仿真平面截面的结果快照。

    参数:
    intensities (dict[str, np.ndarray]): 命名强度分量，形状 (ny, nx)。
    x, y (np.ndarray): 1D 坐标（len(x)=nx 对应轴 1）。
    wavelength (float | None): 单色仿真的波长（多色合成结果为 None）。
    z (float | None): 截面所在平面。
    complex_fields (dict[str, np.ndarray] | None): 复场快照（可选）。
    meta (dict): 自由元数据（元件配置、扫描参数等）。
    """

    intensities: dict[str, np.ndarray]
    x: np.ndarray
    y: np.ndarray
    wavelength: float | None = None
    z: float | None = None
    complex_fields: dict[str, np.ndarray] | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.intensities = {k: np.asarray(v) for k, v in self.intensities.items()}
        if self.complex_fields is not None:
            self.complex_fields = {k: np.asarray(v)
                                   for k, v in self.complex_fields.items()}
        if not self.intensities:
            raise ValueError("intensities 不能为空")
        shape = self._ref_shape
        for name, arr in self.intensities.items():
            if arr.shape != shape:
                raise ValueError(f"强度 '{name}' 形状 {arr.shape} 与参考 {shape} 不一致")
        self.x = np.asarray(self.x)
        self.y = np.asarray(self.y)

    @property
    def _ref_shape(self):
        return next(iter(self.intensities.values())).shape

    # ------------------------------------------------------------------
    @property
    def total_intensity(self) -> np.ndarray:
        """全部强度分量之和（形状 (ny, nx)）。"""
        out = None
        for arr in self.intensities.values():
            out = arr if out is None else out + arr
        return out

    def component(self, name: str) -> np.ndarray:
        """复场分量（无 complex_fields 或缺名时给出明确错误）。"""
        if self.complex_fields is None:
            raise KeyError("该 Field 未保存复场（构造时需提供 complex_fields）")
        return self.complex_fields[name]

    @property
    def dx(self):
        return float(self.x[1] - self.x[0])

    @property
    def dy(self):
        return float(self.y[1] - self.y[0])

    def power(self) -> float:
        """Σ total_intensity · dx · dy。"""
        return float(self.total_intensity.sum()) * self.dx * self.dy

    # ------------------------------------------------------------------
    def line_cut(self, axis='x', position=None) -> dict:
        """过指定位置的截线。

        position=None 时取 total_intensity 峰值所在位置。
        返回 {'coord': 轴坐标, <name>: 强度线, ...}。
        """
        T = self.total_intensity
        if axis == 'x':
            peak_idx = int(np.unravel_index(np.argmax(T), T.shape)[0])
            idx = (peak_idx if position is None
                   else int(np.argmin(np.abs(self.y - position))))
            out = {'coord': self.x}
            for name, arr in self.intensities.items():
                out[name] = arr[idx, :]
            out['total'] = T[idx, :]
        elif axis == 'y':
            peak_idx = int(np.unravel_index(np.argmax(T), T.shape)[1])
            idx = (peak_idx if position is None
                   else int(np.argmin(np.abs(self.x - position))))
            out = {'coord': self.y}
            for name, arr in self.intensities.items():
                out[name] = arr[:, idx]
            out['total'] = T[:, idx]
        else:
            raise ValueError("axis 必须是 'x' 或 'y'")
        return out

    # ------------------------------------------------------------------
    def save(self, path):
        """压缩 npz 持久化（元数据 JSON 编码，数组按前缀分类）。"""
        payload = {
            '__x__': self.x, '__y__': self.y,
            '__meta__': np.array(json.dumps({
                'wavelength': self.wavelength, 'z': self.z, 'meta': self.meta,
                'intensity_names': list(self.intensities),
                'complex_names': (list(self.complex_fields)
                                  if self.complex_fields else None),
            })),
        }
        for name, arr in self.intensities.items():
            payload[f'intensity::{name}'] = arr
        if self.complex_fields:
            for name, arr in self.complex_fields.items():
                payload[f'complex::{name}'] = arr
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **payload)

    @classmethod
    def load(cls, path) -> Field:
        """从 npz 恢复（save 的精确逆操作）。"""
        with np.load(path, allow_pickle=False) as data:
            header = json.loads(str(data['__meta__']))
            intensities = {name: data[f'intensity::{name}']
                           for name in header['intensity_names']}
            complex_fields = None
            if header['complex_names']:
                complex_fields = {name: data[f'complex::{name}']
                                  for name in header['complex_names']}
            return cls(
                intensities=intensities,
                x=data['__x__'], y=data['__y__'],
                wavelength=header['wavelength'], z=header['z'],
                complex_fields=complex_fields, meta=header['meta'],
            )

    # ------------------------------------------------------------------
    @classmethod
    def from_scalar_system(cls, system, z_positions=None, keep_complex=True,
                           **prop_kwargs) -> list[Field]:
        """从标量 OpticalSystem 提取结果（propagate_to_cross_sections 包装）。

        z_positions 默认用系统的最后一个元件位置（或 [0]）；
        单个 float 自动包装成列表。
        """
        z_positions = _normalize_z(system, z_positions)
        results = system.propagate_to_cross_sections(
            list(z_positions), **prop_kwargs)
        fields = []
        for z, payload in results.items():
            U, x, y = payload[0]
            fields.append(cls(
                intensities={'U': np.abs(U) ** 2},
                x=x, y=y, wavelength=system.wavelength, z=float(z),
                complex_fields={'U': U} if keep_complex else None,
                meta={'system': type(system).__name__},
            ))
        return fields

    @classmethod
    def from_vector_system(cls, system, z_positions=None, keep_complex=True,
                           **prop_kwargs) -> list[Field]:
        """从 VectorOpticalSystem 提取结果（三分量强度 + 可选复场）。"""
        z_positions = _normalize_z(system, z_positions)
        results = system.propagate_to_cross_sections(list(z_positions),
                                                     **prop_kwargs)
        fields = []
        for z, vec in results.items():
            intensities = {'ex': np.abs(vec.ex) ** 2,
                           'ey': np.abs(vec.ey) ** 2}
            complex_fields = {'ex': vec.ex, 'ey': vec.ey}
            if vec.ez is not None:
                intensities['ez'] = np.abs(vec.ez) ** 2
                complex_fields['ez'] = vec.ez
            fields.append(cls(
                intensities=intensities, x=vec.x, y=vec.y,
                wavelength=system.wavelength, z=float(z),
                complex_fields=complex_fields if keep_complex else None,
                meta={'system': type(system).__name__},
            ))
        return fields
