"""C3 ParameterSweep：参数扫描执行器与像质指标库。

设计：扫描 = 「对每组参数构建一个全新系统 → 传播到指定平面 →
计算指标」。GPU 仿真本身串行执行（单卡），扫描循环不做并行；
开销集中在传播，回调构建系统的成本可忽略。

指标库（``evaluate_metrics``）在 Field 容器上计算：
- ``peak_intensity``   : 峰值强度；
- ``peak_position``    : 峰值像素坐标 (x, y)；
- ``centroid``         : 归一化质心 (cx, cy)（峰附近 3σ 窗口内，
  避免远场噪声主导）；
- ``fwhm_x`` / ``fwhm_y``: 过峰值截线的半高全宽（线性插值亚像素）。

参数网格用笛卡尔积展开；结果行按参数列 + 指标列扁平存储，
``to_dataframe`` 直接进 pandas（分析/绘图/落 CSV）。
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from dataclasses import field as dc_field

import numpy as np
from tqdm import tqdm

from workflows.field import Field

DEFAULT_METRICS = ('peak_intensity', 'fwhm_x', 'fwhm_y', 'centroid')


# ---------------------------------------------------------------------------
# 指标库
# ---------------------------------------------------------------------------
def _interp_edge(line, coords, i_edge, direction):
    """过阈值的亚像素边缘（线性插值）。direction=+1 右沿 / −1 左沿。"""
    j = i_edge + direction
    if not (0 <= j < len(line)):
        return coords[i_edge]
    y0, y1 = line[j], line[i_edge]
    if y1 == y0:
        return coords[i_edge]
    frac = (line.max() / 2 - y0) / (y1 - y0)
    return coords[j] + frac * (coords[i_edge] - coords[j])


def _fwhm_along(field: Field, axis: str) -> float:
    """过 total_intensity 峰值的截线 FWHM（μm）。"""
    T = field.total_intensity
    if axis == 'x':
        iy = int(np.unravel_index(np.argmax(T), T.shape)[0])
        line, coords = T[iy, :], field.x
    elif axis == 'y':
        ix = int(np.unravel_index(np.argmax(T), T.shape)[1])
        line, coords = T[:, ix], field.y
    else:
        raise ValueError("axis 必须是 'x' 或 'y'")
    half = line.max() / 2
    above = np.flatnonzero(line >= half)
    if above.size == 0:
        return float('nan')
    i_left, i_right = int(above[0]), int(above[-1])
    xl = _interp_edge(line, coords, i_left, -1)
    xr = _interp_edge(line, coords, i_right, +1)
    return float(xr - xl)


def _centroid(field: Field, window_sigma: float = 3.0) -> tuple[float, float]:
    """峰值附近有限窗口内的归一化质心（避免拖尾噪声主导）。"""
    T = field.total_intensity
    ny, nx = T.shape
    iy, ix = np.unravel_index(np.argmax(T), T.shape)
    X, Y = np.meshgrid(field.x, field.y)
    scale = max(field.dx, field.dy) * window_sigma * 2
    mask = ((X - field.x[ix]) ** 2 + (Y - field.y[iy]) ** 2) <= scale ** 2
    w = np.where(mask, T, 0.0)
    total = w.sum()
    if total <= 0:
        return float('nan'), float('nan')
    cx = float((w * X).sum() / total)
    cy = float((w * Y).sum() / total)
    return cx, cy


def evaluate_metrics(field: Field, metrics=DEFAULT_METRICS) -> dict:
    """按名计算指标字典。未知名抛 ValueError。"""
    out = {}
    for name in metrics:
        if name == 'peak_intensity':
            out[name] = float(field.total_intensity.max())
        elif name == 'peak_position':
            iy, ix = np.unravel_index(np.argmax(field.total_intensity),
                                      field.total_intensity.shape)
            out[name] = (float(field.x[ix]), float(field.y[iy]))
        elif name == 'centroid':
            out[name] = _centroid(field)
        elif name == 'fwhm_x':
            out[name] = _fwhm_along(field, 'x')
        elif name == 'fwhm_y':
            out[name] = _fwhm_along(field, 'y')
        else:
            raise ValueError(f"未知指标: {name!r}（可用: {DEFAULT_METRICS}）")
    return out


# ---------------------------------------------------------------------------
# 扫描执行器
# ---------------------------------------------------------------------------
@dataclass
class SweepResult:
    """参数扫描结果：扁平记录行 + 可选保留的 Field 快照。"""

    param_names: tuple[str, ...]
    records: list[dict]
    fields: dict[tuple, Field] = dc_field(default_factory=dict)

    def to_dataframe(self):
        """pandas DataFrame（参数列在前，指标列在后）。"""
        import pandas as pd
        rows = []
        for rec in self.records:
            row = {}
            for name in self.param_names:
                row[name] = rec[name]
            for key, value in rec.items():
                if key in self.param_names:
                    continue
                if isinstance(value, tuple):
                    for i, v in enumerate(value):
                        row[f'{key}[{i}]'] = v
                else:
                    row[key] = value
            rows.append(row)
        return pd.DataFrame(rows)

    def best(self, metric: str, mode='min'):
        """按指标最优（nan 忽略）返回记录行。"""
        key_fn = (lambda r: r[metric]) if mode == 'min' \
            else (lambda r: -r[metric])
        valid = [r for r in self.records
                 if not np.isnan(r.get(metric, float('nan')))]
        if not valid:
            raise ValueError(f"指标 {metric!r} 无有效值")
        return min(valid, key=key_fn)

    def save(self, path):
        self.to_dataframe().to_csv(path, index=False)


class ParameterSweep:
    """参数扫描执行器。

    参数:
    build_system (callable): ``build_system(**params)`` → 全新系统
        （OpticalSystem / VectorOpticalSystem）。**每次调用必须返回
        新实例**（系统有内部状态）。
    metrics (tuple): evaluate_metrics 的指标名列表，或 None 关闭。
    evaluate (callable | None): 自定义指标 ``evaluate(field, **params)
        → dict``；提供时忽略 metrics。
    """

    def __init__(self, build_system, metrics=DEFAULT_METRICS, evaluate=None):
        self.build_system = build_system
        self.metrics = metrics
        self.evaluate = evaluate

    def run(self, param_grid: dict, z_of=None, keep='metrics',
            progress=True) -> SweepResult:
        """执行扫描。

        param_grid: {参数名: 取值列表}（笛卡尔积全展开；所有参数都
            会传给 build_system 与 evaluate）。
        z_of: callable(**params) → float | list | None。从参数组合推导
            传播目标平面（如二维 (λ, z) 扫描时取 z 参数本身）；
            None 时用系统最后元件位置。
        keep: 'metrics' | 'fields' | 'both'。
        """
        if keep not in ('metrics', 'fields', 'both'):
            raise ValueError("keep 必须是 'metrics' | 'fields' | 'both'")
        names = tuple(param_grid)
        combos = list(itertools.product(*(param_grid[n] for n in names)))
        result = SweepResult(param_names=names, records=[], fields={})

        iterator = tqdm(combos, desc='Parameter sweep') if progress else combos
        for combo in iterator:
            params = dict(zip(names, combo))
            system = self.build_system(**params)
            z_positions = z_of(**params) if z_of is not None else None
            fields = self._extract(system, z_positions)
            last = fields[-1]
            if self.evaluate is not None:
                metrics = dict(self.evaluate(last, **params))
            elif self.metrics:
                metrics = evaluate_metrics(last, self.metrics)
            else:
                metrics = {}
            record = {**params, **_flatten(metrics)}
            result.records.append(record)
            if keep in ('fields', 'both'):
                result.fields[combo] = last
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _extract(system, z_positions):
        from optical_system.system import OpticalSystem
        from vector.system import VectorOpticalSystem
        if isinstance(system, VectorOpticalSystem):
            return Field.from_vector_system(system, z_positions)
        if isinstance(system, OpticalSystem):
            return Field.from_scalar_system(system, z_positions)
        raise TypeError(f"build_system 返回了不支持的类型 {type(system)!r}")


def _flatten(metrics: dict) -> dict:
    out = {}
    for key, value in metrics.items():
        if isinstance(value, tuple):
            for i, v in enumerate(value):
                out[f'{key}[{i}]'] = v
        else:
            out[key] = value
    return out
