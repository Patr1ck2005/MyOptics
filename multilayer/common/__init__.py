"""多层膜公共库：物料加载与 TMM。

收敛历史散落在 round1/optimize_transmittance.py、round1/angle_scan_spectra.py、
round2/angle_scan_spectra_variants.py 中的重复实现：
- ``materials`` : nk 数据加载与波长网格插值
- ``tmm``       : N-导纳形式相干 TMM（kx 驱动，s/p 双偏振）
"""
