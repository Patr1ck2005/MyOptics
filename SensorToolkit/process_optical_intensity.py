# process_optical_intensity.py

from pathlib import Path
import logging
from core.optical_intensity_analyzer import OpticalIntensityAnalyzer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 导入标签（可选）
# from labels import labels

def main():
    # 1. 定义输入和输出路径
    # input_dir = Path("./data/3")
    input_dir = Path("./data/CP/CP-1525~1575/1")
    # input_dir = Path("./data/CP/CP-1525~1575/2")
    # input_dir = Path("./data/CP/CP-1525~1575/3")
    # input_dir = Path("./data/CP/comparision-LP-unpatterned-1550")
    output_csv_path = Path("./rsl/optical_intensity_results.csv")

    # 2. 定义裁剪参数（圆形裁剪，固定中心）

    crop_shape_params = {
        'center_row': 0.38,   # 相对坐标
        'center_col': 0.5,   # 相对坐标
        'radius': 0.35,       # 相对半径
        'inner_radius': 0,       # 相对半径
        'shape': 'circle',
        'relative': True     # 使用相对坐标
    }

    # 3. 初始化分析器（可选添加标签）
    analyzer = OpticalIntensityAnalyzer(
        input_dir=input_dir,
        output_csv_path=output_csv_path,
        crop_shape_params=crop_shape_params,
        labels=None  # 或者传入 labels 字典，例如 labels=labels
    )

    # 4. 添加标签（如果有）
    # analyzer.add_label("1550.bmp", "Sample_A")
    # analyzer.add_label("1600.bmp", "Sample_B")
    # 或者在初始化时传入 labels 字典

    # 5. 执行批量处理
    analyzer.process_all()

    # 6. 打印处理结果（可选）
    # print(analyzer.results)

if __name__ == "__main__":
    main()
