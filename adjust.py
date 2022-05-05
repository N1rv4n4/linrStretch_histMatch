# -*- coding:utf-8 -*-
"""
Created by: 80760
Date: 2022.04.12
"""

# import numpy as np
# import cv2
#
# # 加载图片 读取彩色图像归一化且转换为浮点型
# image = cv2.imread(r'F:\to_mk\result\final_data\train_8bit\1.png', cv2.IMREAD_COLOR).astype(np.float32) / 255.0
#
# # 颜色空间转换 BGR转为HLS
# hlsImg = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
#
# # 然后我们需要做两个滑动块，一个调节亮度，一个调节饱和度：
#
# # 滑动条最大值
# MAX_VALUE = 100
# # 滑动条最小值
# MIN_VALUE = 0
#
# # 调节饱和度和亮度的窗口
# cv2.namedWindow("lightness and saturation", cv2.WINDOW_GUI_NORMAL)
#
# # 创建滑动块
# cv2.createTrackbar("lightness", "lightness and saturation", MIN_VALUE, MAX_VALUE, lambda x:x)
# cv2.createTrackbar("saturation", "lightness and saturation", MIN_VALUE, MAX_VALUE, lambda x:x)
#
# # 调节前还需要保存一下原图，所以我们会在内存里复制一个新的变量用于调节图片，然后获得两个滑动块的值，再根据值进行亮度和饱和度的调整：
#
# # 调整饱和度和亮度
# while True:
#     # 复制原图
#     hlsCopy = np.copy(hlsImg)
#     # 得到 lightness 和 saturation 的值
#     lightness = cv2.getTrackbarPos('lightness', 'lightness and saturation')
#     saturation = cv2.getTrackbarPos('saturation', 'lightness and saturation')
#     # 1.调整亮度（线性变换)
#     hlsCopy[:, :, 1] = (1.0 + lightness / float(MAX_VALUE)) * hlsCopy[:, :, 1]
#     hlsCopy[:, :, 1][hlsCopy[:, :, 1] > 1] = 1
#     # 饱和度
#     hlsCopy[:, :, 2] = (1.0 + saturation / float(MAX_VALUE)) * hlsCopy[:, :, 2]
#     hlsCopy[:, :, 2][hlsCopy[:, :, 2] > 1] = 1
#     # HLS2BGR
#     lsImg = cv2.cvtColor(hlsCopy, cv2.COLOR_HLS2BGR)
#     # 显示调整后的效果
#     cv2.imshow("lightness and saturation", lsImg)
#
#     ch = cv2.waitKey(5)
#     # 按 ESC 键退出
#     if ch == 27:
#         break
#     elif ch == ord('s'):
#         # 按 s 键保存并退出
#         lsImg = lsImg * 255
#         lsImg = lsImg.astype(np.uint8)
#         # cv2.imwrite("lsImg.jpg", lsImg)
#         break

#############################################################################################################

import numpy as np
import cv2
from PIL import Image
from matplotlib import pyplot as plt
import pylab
import os

# VisualEffect
class VisualEffect:
    """
    生成从给定间隔均匀采用的视觉效果参数
    参数：
    contrast_factor: 对比度因子：调整对比度的因子区间，应该在0-3之间
    brightness_delta: 亮度增量：添加到像素的量在-1和1之间的间隔
    hue_delta:色度增量：为添加到色调通道的量在-1和1之间的间隔
    saturation_factor:饱和系数：因子乘以每个像素的饱和值的区间
    """
    def __init__(
            self,
            contrast_factor,
            brightness_delta,
            hue_delta,
            saturation_factor
    ):
        self.contrast_factor = contrast_factor
        self.brightness_delta = brightness_delta
        self.hue_delta = hue_delta
        self.saturation_factor = saturation_factor

    def __call__(self, image):
        """
        将视觉效果应用到图片上
        """
        if self.contrast_factor:  # 对比度
            image = self.adjust_contrast(image, self.contrast_factor)
            # print('1')
        if self.brightness_delta:  # 亮度
            image = self.adjust_brightness(image, self.brightness_delta)
            # print('2')
        if self.hue_delta or self.saturation_factor:

            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) #色彩空间转化
            if self.hue_delta:
                image = self.adjust_hue(image, self.hue_delta)
            if self.saturation_factor:
                image = self.adjust_saturation(image, self.saturation_factor)
            # print('3')
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)#色彩空间转化
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def adjust_saturation(self, image, factor):
        """
        调整图片的饱和度
        """
        image[..., 1] = np.clip(image[..., 1] * factor, 0, 255)
        return image

    def adjust_hue(self, image, delta):
        """
        调整图片的色度
        添加到色调通道的量在-1和1之间的间隔。
        如果值超过180，则会旋转这些值。
        """
        image[..., 0] = np.mod(image[...,0] + delta * 180, 180) #取余数
        return image

    def adjust_contrast(self, image, factor):
        """
        调整一张图像的对比度
        """
        mean = image.mean(axis=0).mean(axis=0)
        # print(factor)
        return self._clip((image - mean) * factor + mean)

    def adjust_brightness(self, image, delta):
        """
        调整一张图片的亮度
        """
        # print(delta)
        image1 = self._clip(image + delta * 255)
        h, w, c = image1.shape
        while np.sum(image1 == 0) > 0.5 * h * w * c:
            delta = delta + 0.1
            # print('------------------------------------------------------------')
            image1 = self._clip(image + delta * 255)
            print('all zero')
            # print('---------------------------------------------------------------')
        return image1

    def _clip(self,image):
        """
        剪辑图像并将其转换为np.uint8
        """
        return np.clip(image, 0, 255).astype(np.uint8)


def _uniform(val_range):
    """
    随机返回值域之间的数值
    """
    return np.random.uniform(val_range[0], val_range[1])


def _check_range(val_range, min_val=None, max_val=None):
    """
    检查间隔是否有效
    """
    if val_range[0] > val_range[1]:
        raise ValueError('interval lower bound > upper bound')
    if min_val is not None and val_range[0] < min_val:
        raise ValueError('invalid interval lower bound')
    if max_val is not None and val_range[1] > max_val:
        raise ValueError('invalid interval upper bound')


# 定义随机视觉效果生成器
def random_visual_effect_generator(
        contrast_range=(0.9, 1.1),
        brightness_range=(-.1, .1),
        hue_range=(-0.05, 0.05),
        saturation_range=(0.95, 1.05)):
    _check_range(contrast_range, 0)
    _check_range(brightness_range, -1, 1)
    _check_range(hue_range, -1, 1)
    _check_range(saturation_range, 0)

    def _generate():
        while True:
            yield VisualEffect(
                contrast_factor=_uniform(contrast_range),
                brightness_delta=_uniform(brightness_range),
                hue_delta=_uniform(hue_range),
                saturation_factor=_uniform(saturation_range)
            )

    return _generate()

def checkAndMakeDir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print('path exists')
    else:
        print('path exists')

if __name__ == "__main__":
    path = r"F:\shipDetecData3\Input"
    savepath = r"F:\shipDetecData3\weakInput"
    checkAndMakeDir(savepath)
    imagedir = os.listdir(path)

    visual_effect_generator = random_visual_effect_generator(
        contrast_range=(0.3, 0.5),  # 最小值0，最大值不限  0.01 0.3
        brightness_range=(-0.1, 0),  # 最小值-1，最大值1
        hue_range=(0, 0),
        saturation_range=(0, 0)
    )  # 创建生成器

    for filename in imagedir:
        img_path = path + '/' + filename
        image = np.asarray(Image.open(img_path))

        visual_effect = next(visual_effect_generator)
        imageOut = visual_effect(image)
        # print(image)
        # plt.imshow(image)
        # pylab.show()

        imageOut = cv2.cvtColor(imageOut, cv2.COLOR_BGR2RGB)
        cv2.imwrite(savepath + '/' + filename, imageOut)
