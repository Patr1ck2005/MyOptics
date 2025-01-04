# Sensor Data Processing Toolkit

## 项目简介

Sensor Data Processing Toolkit 是一个用于处理传感器数据的 Python 工具包。该工具包支持从 CSV 文件和多种图像格式（如 BMP、PNG、JPG 等）加载数据，执行数据裁剪、坐标重置、计算平均光强等处理操作，并提供数据的可视化和保存功能。通过模块化和面向对象的设计，工具包具备良好的可扩展性和可维护性，适用于科研、工程及数据分析等领域。

## 功能概述

- **数据加载**：
  - 从 CSV 文件加载传感器数据。
  - 从图像文件（支持 BMP、PNG、JPG、JPEG、TIFF 等格式）加载传感器数据。

- **数据处理**：
  - 数据裁剪：支持按形状（圆形、方形）和比例或绝对像素裁剪数据。
  - 坐标重置：重置数据的行列坐标。
  - 平均光强计算：计算裁剪后数据的平均光强。
  - 数据归一化：将数据缩放到 0-1 范围。

- **数据可视化**：
  - 可视化处理后的数据，支持显示和保存图像。

- **数据保存**：
  - 将处理后的数据保存为 CSV 文件。
  - 将处理后的数据保存为图像文件（灰度图像）。

- **批量处理**：
  - 支持对工作目录中的多个图像文件进行批量处理。

## 安装指南

### 前提条件

- Python 3.7 及以上版本
- pip 包管理工具

### 安装依赖

1. **克隆项目仓库**（如果适用）：

   ```bash
   git clone https://github.com/yourusername/sensor-data-processing-toolkit.git
   cd sensor-data-processing-toolkit
   ```

2. **创建虚拟环境**（可选，但推荐）：

   ```bash
   python -m venv venv
   source venv/bin/activate  # 在 Windows 上使用 `venv\Scripts\activate`
   ```

3. **安装必要的 Python 库**：

   创建一个 `requirements.txt` 文件，内容如下：

   ```txt
   pandas
   numpy
   matplotlib
   Pillow
   ```

   然后运行：

   ```bash
   pip install -r requirements.txt
   ```

## 使用指南

### 项目结构

```
sensor-data-processing-toolkit/
├── data/
│   ├── bmp/               # 输入的 BMP 图像文件目录
│   ├── processed_csv/    # 输出的处理后 CSV 文件目录
│   └── ...                # 其他数据目录
├── sensor_data_processor.py  # 主处理脚本
├── README.md
└── requirements.txt
```

### 类与模块说明

#### 1. `DataLoader` (抽象基类)

- **职责**：定义数据加载的接口。
- **子类**：
  - `CSVDataLoader`：用于加载 CSV 文件。
  - `ImageDataLoader`：用于加载图像文件（支持 BMP、PNG、JPG、JPEG、TIFF 等格式）。

#### 2. `DataProcessor`

- **职责**：处理加载的数据，包括裁剪、坐标重置、计算平均光强和归一化。
- **方法**：
  - `crop_by_shape()`: 按指定形状（圆形或方形）裁剪数据。
  - `reset_coordinates()`: 重置数据的行列坐标。
  - `calculate_average_intensity()`: 计算数据的平均光强。
  - `rescale()`: 将数据重新缩放到 0-1 范围。

#### 3. `Visualizer`

- **职责**：负责数据的可视化展示。
- **方法**：
  - `visualize()`: 可视化数据，支持显示和保存图像。

#### 4. `DataSaver` (抽象基类)

- **职责**：定义数据保存的接口。
- **子类**：
  - `CSVSaver`：用于将数据保存为 CSV 文件。
  - `ImageSaver`：用于将数据保存为图像文件（灰度图像）。

#### 5. `SensorDataProcessor`

- **职责**：整合数据加载、处理、可视化和保存的流程。
- **方法**：
  - `process()`: 处理数据，包括裁剪、重置坐标、计算平均光强和归一化。
  - `visualize()`: 可视化处理后的数据。
  - `save()`: 保存处理后的数据。

#### 6. `BatchProcessor`

- **职责**：负责批量处理输入目录中的所有 BMP 文件。
- **方法**：
  - `process_all()`: 批量处理所有 BMP 文件，包括加载、裁剪、计算平均光强和保存。

### 示例代码

以下是一个使用工具包处理单个 BMP 文件的示例：

```python
from pathlib import Path
import logging
from sensor_data_processor import (
    ImageDataLoader,
    DataProcessor,
    Visualizer,
    CSVSaver,
    SensorDataProcessor
)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. 定义输入和输出路径
image_path = Path("./data/bmp/1550-Au-fake.bmp")
output_csv_path = Path("./data/processed_csv/1550_Au_faked.csv")

# 2. 初始化数据加载器
loader = ImageDataLoader(image_path)

# 3. 初始化数据处理器
processor = DataProcessor(data=pd.DataFrame())  # 初始时传入空 DataFrame，稍后由 SensorDataProcessor 填充

# 4. 初始化可视化器和保存器
visualizer = Visualizer()
saver = CSVSaver()

# 5. 初始化传感器数据处理器
sensor_processor = SensorDataProcessor(loader, processor, visualizer, saver)

# 6. 定义裁剪参数
crop_shape_params = {
    'center_row': 0.5,   # 相对坐标，50%
    'center_col': 0.5,   # 相对坐标，50%
    'radius': 0.3,       # 相对半径，30%
    'shape': 'circle',
    'relative': True     # 使用相对坐标
}

# 7. 处理数据
avg_intensity = sensor_processor.process(crop_shape_params=crop_shape_params)
logging.info(f"平均光强: {avg_intensity}")

# 8. 可视化处理后的数据
sensor_processor.visualize(title="Processed Sensor Data")

# 9. 保存处理后的数据为 CSV
sensor_processor.save(output_csv_path)
```

### 批量处理多个文件

以下是批量处理工作目录中所有 BMP 文件的示例：

```python
from pathlib import Path
import logging
from sensor_data_processor import (
    ImageDataLoader,
    DataProcessor,
    Visualizer,
    CSVSaver,
    BatchProcessor
)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. 定义输入和输出目录
input_dir = Path("./data/bmp")
output_dir = Path("./data/processed_csv")

# 2. 定义裁剪参数
crop_shape_params = {
    'center_row': 0.5,   # 相对坐标，50%
    'center_col': 0.5,   # 相对坐标，50%
    'radius': 0.3,       # 相对半径，30%
    'shape': 'circle',
    'relative': True     # 使用相对坐标
}

# 3. 初始化批处理器
batch_processor = BatchProcessor(input_dir, output_dir, crop_shape_params)

# 4. 执行批量处理
batch_processor.process_all()
```

## 使用说明

### 步骤 1：设置工作目录

确保您的工作目录包含以下结构：

```
your_working_directory/
├── data/
│   ├── bmp/
│   │   ├── 1550-Au-fake.bmp
│   │   ├── 1600-Au-fake.bmp
│   │   └── ... (其他波长的 BMP 文件)
│   └── processed_csv/  # 处理后的 CSV 文件将保存在这里
├── sensor_data_processor.py  # 主处理脚本
├── README.md
└── requirements.txt
```

### 步骤 2：准备数据文件

将所有需要处理的 BMP 图像文件放置在 `data/bmp/` 目录下。确保文件命名格式为 `{wavelength}-*.bmp`，例如 `1550-Au-fake.bmp`，其中 `1550` 表示波长（单位为 nm）。

### 步骤 3：运行处理脚本

使用示例代码中的批量处理示例，或者根据需要编写自定义脚本来处理数据。

### 步骤 4：查看处理结果

处理后的 CSV 文件将保存在 `data/processed_csv/` 目录中，文件名与原始图像文件的波长对应，例如 `1550_Au_faked.csv`。

## 日志记录

工具包使用 Python 的 `logging` 模块记录处理过程中的信息，包括：

- 成功加载文件的信息。
- 裁剪参数和裁剪区域。
- 计算的平均光强。
- 保存文件的位置。
- 处理过程中发生的错误信息。

日志信息将输出到控制台，便于实时监控处理进度。

## 错误处理

- **文件不存在**：如果指定的输入文件或目录不存在，程序将抛出 `FileNotFoundError` 并记录错误信息。
- **不支持的文件格式**：如果尝试加载不支持的图像格式，程序将抛出 `ValueError` 并记录错误信息。
- **数据处理错误**：在数据处理过程中，如果发生错误（如除以零），程序将记录错误信息并抛出异常。

即使在批量处理过程中遇到单个文件的错误，程序也会继续处理其他文件，确保整体流程的稳定性。

## 扩展与自定义

- **支持更多文件格式**：可以在 `ImageDataLoader` 类的 `SUPPORTED_FORMATS` 集合中添加更多支持的图像格式。
- **添加更多数据处理功能**：如滤波、边缘检测等，可以在 `DataProcessor` 类中添加相应的方法。
- **自定义可视化**：可以修改 `Visualizer` 类中的 `visualize` 方法，以适应不同的可视化需求。

## 贡献

欢迎对 Sensor Data Processing Toolkit 进行贡献！请按照以下步骤操作：

1. Fork 本仓库。
2. 创建新分支 (`git checkout -b feature/YourFeature`)。
3. 提交更改 (`git commit -m 'Add some feature'`)。
4. 推送到分支 (`git push origin feature/YourFeature`)。
5. 创建 Pull Request。

## 许可证

本项目采用 MIT 许可证。详情请参阅 [LICENSE](LICENSE)。

## 联系方式

如有任何问题或建议，请联系 [your.email@example.com](mailto:your.email@example.com)。

---

感谢您使用 Sensor Data Processing Toolkit！希望它能为您的数据处理工作带来便利。