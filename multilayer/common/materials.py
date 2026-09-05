"""物料 nk 数据加载与插值。

数据源约定（round1/round2 共用同一格式）：
- Ag.csv / ZnO.csv : 两段式 CSV（"wl,n" 段与 "wl,k" 段）
- 薄膜.xlsx        : 三列 (wl, n, k)

数据目录须显式传入（round1/round2 各有一份），无全局默认。
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def load_csv_two_section(filepath: Path):
    """Load a material CSV containing separate wl,n and wl,k sections."""
    lines = filepath.read_text(encoding="utf-8").splitlines()
    section_starts = {}
    for index, line in enumerate(lines):
        header = line.strip().lower()
        if header.startswith("wl,n"):
            section_starts["n"] = index + 1
        elif header.startswith("wl,k"):
            section_starts["k"] = index + 1

    if "n" not in section_starts or "k" not in section_starts:
        raise ValueError(f"Could not find wl,n and wl,k sections in {filepath}")

    def read_section(start):
        wavelengths = []
        values = []
        for line in lines[start:]:
            text = line.strip()
            if not text or text.lower().startswith("wl"):
                break
            parts = text.split(",")
            if len(parts) < 2:
                continue
            wavelengths.append(float(parts[0]))
            values.append(float(parts[1]))
        return np.asarray(wavelengths, dtype=float), np.asarray(values, dtype=float)

    return read_section(section_starts["n"]), read_section(section_starts["k"])


def load_material(name: str, data_dir: Path):
    """Return raw wavelength, n, wavelength, k arrays for one material."""
    if name in {"Ag", "ZnO"}:
        (wl_n, n), (wl_k, k) = load_csv_two_section(data_dir / f"{name}.csv")
        return wl_n, n, wl_k, k
    if name == "Film":
        values = pd.read_excel(data_dir / "薄膜.xlsx", header=None).apply(
            pd.to_numeric, errors="coerce"
        ).dropna().to_numpy(dtype=float)
        if values.shape[1] < 3:
            raise ValueError("Film workbook must contain wavelength, n, and k columns")
        wl_n = values[:, 0]
        n = values[:, 1]
        k = values[:, 2]
        return wl_n, n, wl_n.copy(), k
    raise ValueError(f"Unknown material: {name}")


def build_nk_interpolators(name: str, wl_grid, data_dir: Path):
    """Build n(λ) and k(λ) interpolators on a common wavelength grid (in μm).

    超出测量范围的波长用最近邻外推（与历史实现一致）。
    """
    wl_n, n_raw, wl_k, k_raw = load_material(name, data_dir)

    n_interp = interp1d(wl_n, n_raw, kind="linear",
                        bounds_error=False, fill_value=(n_raw[0], n_raw[-1]))
    k_interp = interp1d(wl_k, k_raw, kind="linear",
                        bounds_error=False, fill_value=(k_raw[0], k_raw[-1]))

    return n_interp(wl_grid), k_interp(wl_grid)
