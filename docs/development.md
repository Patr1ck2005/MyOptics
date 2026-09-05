# Development

开发环境与流程说明（配合 [CI 与自托管 runner](ci.md) 使用）。

## 本地开发

```bash
pip install -e .[dev]       # pytest + ruff
ruff check .
python -m pytest tests/ -m "not gpu"   # 快速
python -m pytest tests/                # 全量
```

## 测试标记

| marker | 含义 | CI |
|--------|------|-----|
| `gpu` | 需要 CUDA GPU 内核链 | 自托管 runner 执行 `-m gpu` |
| `slow` | 长耗时性能对照 | 默认包含，防退化阈值宽松 |

## 黄金图像基准更新

改变渲染输出（配色/布局/物理）后：

```bash
rm tests/golden/*.npy
python -m pytest tests/test_golden_images.py    # 重新生成基准
git diff tests/golden/                          # 审查哈希变化
```

## 文档站

```bash
mkdocs serve          # 本地预览
mkdocs gh-deploy      # 发布到 GitHub Pages
```

## 版本与发布

- 每个里程碑打 tag：v0.3.0 (M1) → v0.4.0 (M2 矢量引擎) → v0.5.0 (M3 宽带批量)
- CHANGELOG.md 逐项记录；commit 需带 `Origin:` provenance trailer
