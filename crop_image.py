# -*- coding:utf-8 -*-
"""
Created by: 80760
Date: 2022.03.29
"""

import numpy as np
import cv2 as cv
from osgeo import gdal, gdal_array
import glob
import os
from PIL import Image
import sys
import csv
from adjust_image import checkAndMakeDir

# 裁剪图片

def fill_image(image):
    # width, height = image.size
    # 选取长和宽中较大值作为新图片的
    # new_image_length = width if width > height else height
    new_image_length = 4096
    # 生成新图片[白底]
    new_image = Image.new(image.mode, (new_image_length, new_image_length), color='white')
    # #将之前的图粘贴在新图上，居中
    # if width > height:#原图宽大于高，则填充图片的竖直维度
    #     #(x,y)二元组表示粘贴上图相对下图的起始位置
    #     new_image.paste(image, (0, int((new_image_length - height) / 2)))
    # else:
    #     new_image.paste(image,(int((new_image_length - width) / 2),0))
    new_image.paste(image)

    return new_image

#切图
def cut_image(image):
    width, height = image.size
    item_width = int(width / 2)
    box_list = []
    # (left, upper, right, lower)
    for i in range(0, 2):#两重循环，生成9张图片基于原图的位置
        for j in range(0, 2):
            #print((i*item_width,j*item_width,(i+1)*item_width,(j+1)*item_width))
            box = (j*item_width, i*item_width, (j+1)*item_width, (i+1)*item_width)
            box_list.append(box)
    print(box_list)
    image_list = [image.crop(box) for box in box_list]
    return image_list


# 保存
def save_images(image_list, imageOriginalName,savePath):
    index = 1
    str1 = imageOriginalName
    str2 = '.'
    dotIndex = str1.index(str2)
    imageOriginalName = imageOriginalName[:dotIndex]
    for image in image_list:
        image.save(savePath + '/' + imageOriginalName + '_' + str(index) + '.tif')
        index += 1


if __name__ == '__main__':
    path = r'F:\shipDetecData4\final_real_dataset_onlypositivesamles\images'
    savePath = r'F:\shipDetecData4\final_real_dataset_onlypositivesamles\images_cut'
    # tif_list = glob.glob((path + '/*.tif'))
    checkAndMakeDir(savePath)
    datalist = os.listdir(path)

    for dataname in datalist:
        # print(path + '/' + dataname)
        dataset= gdal.Open(path + '/' + dataname)
        datatype = dataset.GetRasterBand(1).DataType
        image = np.zeros((dataset.RasterYSize, dataset.RasterXSize, dataset.RasterCount),
                         dtype=gdal_array.GDALTypeCodeToNumericTypeCode(datatype))

        for b in range(dataset.RasterCount):
            band = dataset.GetRasterBand(b + 1)  # tiff图像的通道数是从 1 开始的
            # ReadAsArray(<xoff>, <yoff>, <xsize>, <ysize>)，读出从(xoff,yoff)开始，大小为(xsize,ysize)的矩阵。
            image[:, :, b] = band.ReadAsArray()

        # opencv（nparray）转为 PIL
        image = Image.fromarray(cv.cvtColor(image, cv.COLOR_BGR2RGB))

        # 图片扩展为正方形
        # image = fill_image(image)

        # 裁剪并保存
        image_list = cut_image(image)
        # print(image.size)
        save_images(image_list, dataname, savePath)
