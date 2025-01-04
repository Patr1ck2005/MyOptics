import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from tkinter import Tk, Frame, Label, Button, DoubleVar, Scale, HORIZONTAL
from PIL import Image

from exp_postpocesses.test.phase_extract_cal import calculate_fourier, apply_filter


class FourierApp:
    def __init__(self, root, image_path):
        self.root = root
        self.root.title("Fourier Transform Visualization")

        # 加载图像并计算初始傅里叶变换
        img = Image.open(image_path)
        img = img.convert('L')  # 转换为灰度图像
        self.image = np.array(img)
        self.F = calculate_fourier(self.image)

        self.ny, self.nx = self.image.shape

        # 初始化参数
        self.loc_x = DoubleVar(value=self.nx // 2)
        self.loc_y = DoubleVar(value=self.ny // 2)
        self.radius = DoubleVar(value=10)

        # 创建控制面板
        control_frame = Frame(root)
        control_frame.grid(row=0, column=0, padx=10, pady=10)

        # loc_x 控制
        Label(control_frame, text="loc_x").grid(row=0, column=0)
        self.loc_x_scale = Scale(control_frame, from_=0, to=self.nx, orient=HORIZONTAL, variable=self.loc_x, command=self.update_canvas)
        self.loc_x_scale.grid(row=0, column=1, columnspan=3)
        Button(control_frame, text="+", command=lambda: self.adjust_param(self.loc_x, 1)).grid(row=0, column=4)
        Button(control_frame, text="-", command=lambda: self.adjust_param(self.loc_x, -1)).grid(row=0, column=5)

        # loc_y 控制
        Label(control_frame, text="loc_y").grid(row=1, column=0)
        self.loc_y_scale = Scale(control_frame, from_=0, to=self.ny, orient=HORIZONTAL, variable=self.loc_y, command=self.update_canvas)
        self.loc_y_scale.grid(row=1, column=1, columnspan=3)
        Button(control_frame, text="+", command=lambda: self.adjust_param(self.loc_y, 1)).grid(row=1, column=4)
        Button(control_frame, text="-", command=lambda: self.adjust_param(self.loc_y, -1)).grid(row=1, column=5)

        # radius 控制
        Label(control_frame, text="radius").grid(row=2, column=0)
        self.radius_scale = Scale(control_frame, from_=1, to=50, orient=HORIZONTAL, variable=self.radius, command=self.update_canvas)
        self.radius_scale.grid(row=2, column=1, columnspan=3)
        Button(control_frame, text="+", command=lambda: self.adjust_param(self.radius, 1)).grid(row=2, column=4)
        Button(control_frame, text="-", command=lambda: self.adjust_param(self.radius, -1)).grid(row=2, column=5)

        # 初始化 Matplotlib 图形
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=1, column=0)

        # 添加放大工具
        toolbar_frame = Frame(root)
        toolbar_frame.grid(row=2, column=0, pady=5)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()

        # 初始化显示
        self.update_canvas()

    def adjust_param(self, param, step):
        """
        微调参数值，并同步更新滑动条。
        """
        param.set(param.get() + step)
        self.update_canvas()

    def update_canvas(self, *args):
        """
        更新画布内容，包括傅里叶谱和提取后的图像。
        """
        loc_x = int(self.loc_x.get())
        loc_y = int(self.loc_y.get())
        radius = int(self.radius.get())

        # 计算滤波结果
        F_filtered, interference_filtered = apply_filter(self.F, loc_x, loc_y, radius, self.ny, self.nx)

        # 更新傅里叶谱
        self.axes[0, 0].clear()
        self.axes[0, 0].imshow(np.log(np.abs(self.F) + 1), cmap='gray')
        self.axes[0, 0].set_title('Original Fourier Spectrum')
        self.axes[0, 0].axis('off')

        self.axes[0, 1].clear()
        self.axes[0, 1].imshow(np.log(np.abs(F_filtered) + 1), cmap='gray')
        self.axes[0, 1].set_title('Filtered Fourier Spectrum')
        self.axes[0, 1].axis('off')

        # 更新提取的图像
        self.axes[1, 0].clear()
        self.axes[1, 0].imshow(np.abs(interference_filtered) ** 2, cmap='gray')
        self.axes[1, 0].set_title('Extracted Intensity')
        self.axes[1, 0].axis('off')

        self.axes[1, 1].clear()
        self.axes[1, 1].imshow(np.angle(interference_filtered), cmap='twilight')
        self.axes[1, 1].set_title('Extracted Phase')
        self.axes[1, 1].axis('off')

        self.canvas.draw()



if __name__ == "__main__":
    root = Tk()
    app = FourierApp(root, image_path='./artificial_pattern.png')
    # app = FourierApp(root, image_path='./interference_1.bmp')
    # app = FourierApp(root, image_path='./interference_2.bmp')
    # app = FourierApp(root, image_path='./interference_3.bmp')
    root.mainloop()
