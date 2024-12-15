import cupy as cp

def bilinear_interpolation_gpu(x, y, x0, y0, values):
    """
    在GPU上实现双线性插值。

    参数:
        x, y: (cupy.ndarray) 目标插值的网格点。
        x0, y0: (cupy.ndarray) 已知网格点。
        values: (cupy.ndarray) 在 (x0, y0) 上定义的值。

    返回:
        interp_values: (cupy.ndarray) 插值后的结果。
    """
    # 定位到最近的四个网格点
    x_idx = cp.searchsorted(x0, x) - 1
    y_idx = cp.searchsorted(y0, y) - 1

    x_idx = cp.clip(x_idx, 0, x0.size - 2)
    y_idx = cp.clip(y_idx, 0, y0.size - 2)

    # 计算插值权重
    x1, x2 = x0[x_idx], x0[x_idx + 1]
    y1, y2 = y0[y_idx], y0[y_idx + 1]

    Q11 = values[y_idx, x_idx]
    Q21 = values[y_idx, x_idx + 1]
    Q12 = values[y_idx + 1, x_idx]
    Q22 = values[y_idx + 1, x_idx + 1]

    wx = (x - x1) / (x2 - x1 + 1e-10)
    wy = (y - y1) / (y2 - y1 + 1e-10)

    interp_values = (
            Q11 * (1 - wx) * (1 - wy) +
            Q21 * wx * (1 - wy) +
            Q12 * (1 - wx) * wy +
            Q22 * wx * wy
    )
    return interp_values