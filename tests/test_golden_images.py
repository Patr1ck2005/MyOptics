"""黄金图像回归：固定输入 → 绘图输出 → 感知哈希（dHash）比对。

目的：捕获绘图逻辑与底层物理结果的"静默漂移"——任何影响渲染输出的
改动（配色、布局、数据）都会改变哈希；此时运行 `pytest --golden-update`
重新生成基准，并在 git diff 中人工审查变化是否为预期。

实现：dHash（difference hash）9x8 灰度差分 → 64 bit，汉明距离阈值 6。
该哈希对渲染后端的抗锯齿/字体微差稳健，但对色带/布局/数据形态敏感。
基准哈希存于 tests/golden/cross_sections.npy 并纳入版本控制；
本测试在基准缺失时自动生成并标 skip，保证首次可运行。
"""
import os

import numpy as np
import pytest
from conftest import GPU_OK

pytestmark = pytest.mark.gpu

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")
HAMMING_THRESHOLD = 6


def _dhash(image_gray, hash_size=8):
    """差分哈希：缩放到 (hash_size+1) x hash_size，横向梯度 → bit。"""
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(4, 4), dpi=32)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image_gray, cmap="gray", aspect="auto")
    ax.axis("off")
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].mean(axis=2)
    plt.close(fig)

    from PIL import Image
    img = Image.fromarray(buf.astype(np.uint8)).resize((hash_size + 1, hash_size))
    px = np.asarray(img, dtype=int)
    diff = px[:, 1:] > px[:, :-1]
    return diff.flatten()


def _hamming(a, b):
    return int(np.count_nonzero(a != b))


def _render_cross_sections(tmp_path):
    """固定高斯场 + 缓存传播器的 cross-sections 渲染（唯一数据源）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from optical_system.elements import CircularAperture, Lens
    from optical_system.system import OpticalSystem
    from visualization.plotter import Plotter

    wl, f = 0.8, 50.0
    x = np.linspace(-15, 15, 257)
    X, Y = np.meshgrid(x, x)
    U0 = np.exp(-(X ** 2 + Y ** 2) / 12.0 ** 2).astype(np.complex128)
    system = OpticalSystem(wl, x, x, U0)
    system.add_element(CircularAperture(0, radius=8.0))
    system.add_element(Lens(f / 2, focal_length=f, NA=0.3))
    results = system.propagate_to_cross_sections([0.0, f / 2 + f], propagation_mode='Rigorous')

    plotter = Plotter(x, x, output_dir=str(tmp_path))
    plotter.plot_cross_sections(results, save_label='golden', show=False)
    png = os.path.join(str(tmp_path), 'golden-cross_sections.png')

    img = plt.imread(png)
    plt.close('all')
    return img[..., :3].mean(axis=2)


@pytest.mark.skipif(not GPU_OK, reason="GPU 不可用")
def test_golden_cross_sections(tmp_path, pytestconfig):
    """cross-sections 渲染与黄金基准的 dHash 汉明距离 ≤ 阈值。"""
    gray = _render_cross_sections(tmp_path)
    h = _dhash(gray)

    golden_path = os.path.join(GOLDEN_DIR, "cross_sections.npy")
    if pytestconfig.getoption("--golden-update"):
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        np.save(golden_path, h)
        pytest.skip("已更新黄金基准（--golden-update）；请用 git diff 审查")

    if not os.path.exists(golden_path):
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        np.save(golden_path, h)
        pytest.skip("黄金基准不存在，已生成本次哈希作为基准；请检查后提交")

    golden = np.load(golden_path)
    dist = _hamming(h, golden)
    assert dist <= HAMMING_THRESHOLD, (
        f"渲染输出漂移：dHash 汉明距离 {dist} > {HAMMING_THRESHOLD}。"
        "若为预期改动，请运行 pytest --golden-update 更新基准并审查 diff。")


@pytest.mark.skipif(not GPU_OK, reason="GPU 不可用")
def test_golden_deterministic_same_input_same_hash(tmp_path):
    """同输入两次渲染哈希必须完全一致（排除随机性）。"""
    g1 = _render_cross_sections(tmp_path)
    g2 = _render_cross_sections(tmp_path)
    assert np.array_equal(_dhash(g1), _dhash(g2))
