"""A3 SimSpectrum：宽谱照明的波长分解仿真与合成。

物理模型：宽谱场按波长分解为一系列单色仿真；每个单色分量独立
传播（衍射长度 ∝ λ，材料色散进入折射率与相位元件），最后按光源
光谱权重合成：

- **非相干合成**（intensity=True，默认）：I_total = Σ_s w_s·I_s。
  适用：宽谱光源相干长度远小于系统中最大光程差（LED、荧光、
  太阳光）；各波长分量互不相干，强度直接相加。
- **相干合成**（intensity=False）：E_total = Σ_s √w_s·E_s·e^{iφ_s}，
  复振幅叠加（φ_s 为可配置的相位因子，默认 0 = 等光程假设）。
  适用：各波长分量的光程差小于相干长度（如同一激光器的多纵模、
  超短脉冲的载波包络近似）。注意：跨波长的相位 φ_s = k_s·L 是否
  有效取决于系统光程差建模精度，使用者需自行判断物理适用性。

权重约定：weights 为光谱功率占比，内部归一到 Σw = 1；未提供时取
均匀权重。

组件来源（二选一）：
- ``simulate``: ``simulate(λ) → Field``（每次调用全新仿真，λ 为唯一
  必需参数；可含色散元件）；
- 直接传入已算好的 ``{λ: Field}`` 字典（复用缓存 / 外部计算）。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from workflows.field import Field


@dataclass
class SpectrumResult:
    """宽谱合成结果 + 各波长分量。"""

    wavelengths: np.ndarray          # (S,)
    weights: np.ndarray              # (S,) 已归一
    field: Field                     # 合成后的 Field（wavelength=None）
    components: dict[float, Field]   # {λ: 单色 Field}

    @property
    def spectral_intensity(self) -> np.ndarray:
        """各波长分量的总功率 (S,)（分析光谱传输效率用）。"""
        return np.array([f.power() for f in
                         (self.components[float(w)]
                          for w in self.wavelengths)])


class SimSpectrum:
    """宽谱仿真编排器。

    参数:
    wavelengths (array): 波长列表（与网格坐标同单位）。
    weights (array | None): 光谱功率权重（自动归一；None=均匀）。
    simulate (callable | None): ``simulate(λ) → Field``。
    components (dict | None): 预计算的 {λ: Field}（与 simulate 二选一）。
    """

    def __init__(self, wavelengths, weights=None, simulate=None,
                 components=None):
        self.wavelengths = np.asarray(wavelengths, dtype=float)
        if self.wavelengths.size == 0:
            raise ValueError("wavelengths 不能为空")
        if weights is None:
            weights = np.ones_like(self.wavelengths)
        weights = np.asarray(weights, dtype=float)
        if weights.shape != self.wavelengths.shape:
            raise ValueError("weights 形状须与 wavelengths 一致")
        self.weights = weights / weights.sum()

        if (simulate is None) == (components is None):
            raise ValueError("simulate 与 components 必须二选一")
        self.simulate = simulate
        self._components = None if components is None else {
            float(w): f for w, f in components.items()}

    # ------------------------------------------------------------------
    def _get_components(self, progress) -> dict[float, Field]:
        if self._components is not None:
            return self._components
        comps = {}
        iterator = (tqdm(self.wavelengths, desc='Spectral components')
                    if progress else self.wavelengths)
        for wl in iterator:
            comps[float(wl)] = self.simulate(float(wl))
        return comps

    def run(self, coherent=False, phases=None, progress=True) -> SpectrumResult:
        """执行宽谱仿真并合成。

        coherent (bool): False → 非相干强度合成；True → 相干复振幅合成。
        phases (array | None): 相干合成时各分量的相位因子 e^{iφ_s}
            （默认全 0 = 等光程假设）。
        """
        comps = self._get_components(progress)
        ref = comps[float(self.wavelengths[0])]
        shape = ref._ref_shape
        total_I = None
        total_E = {}

        if coherent:
            phase_factors = (np.ones_like(self.weights) if phases is None
                             else np.asarray(phases))
            if phase_factors.shape != self.weights.shape:
                raise ValueError("phases 形状须与 wavelengths 一致")

        if coherent:
            for s, (wl, weight) in enumerate(zip(self.wavelengths,
                                                 self.weights)):
                comp = comps[float(wl)]
                if comp._ref_shape != shape:
                    raise ValueError(f"λ={wl} 分量的网格形状与参考不一致")
                for name in comp.intensities:
                    E = comp.complex_fields[name]
                    weighted = np.sqrt(weight) * phase_factors[s] * E
                    total_E[name] = (weighted if name not in total_E
                                     else total_E[name] + weighted)
        else:
            for wl, weight in zip(self.wavelengths, self.weights):
                comp = comps[float(wl)]
                if comp._ref_shape != shape:
                    raise ValueError(f"λ={wl} 分量的网格形状与参考不一致")
                part = weight * comp.total_intensity
                total_I = part if total_I is None else total_I + part

        if coherent:
            synth = Field(
                intensities={name: np.abs(E) ** 2 for name, E in total_E.items()},
                x=ref.x, y=ref.y, wavelength=None, z=ref.z,
                complex_fields=total_E,
                meta={'synthesis': 'coherent',
                      'wavelengths': self.wavelengths.tolist(),
                      'weights': self.weights.tolist()},
            )
        else:
            synth = Field(
                intensities={'broadband': total_I},
                x=ref.x, y=ref.y, wavelength=None, z=ref.z,
                complex_fields=None,
                meta={'synthesis': 'incoherent',
                      'wavelengths': self.wavelengths.tolist(),
                      'weights': self.weights.tolist()},
            )
        return SpectrumResult(
            wavelengths=self.wavelengths, weights=self.weights,
            field=synth, components=comps,
        )
