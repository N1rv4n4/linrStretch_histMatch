# -*- coding:utf-8 -*-
"""
Created by: 80760
Date: 2021.11.19
"""

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']   # 用黑体显示中文


# 将灰度数组映射为直方图字典,nums表示灰度的数量级
def arrayToHist(grayArray, nums):
    if(len(grayArray.shape) != 2):
        print("length error")
        return None
    w,h = grayArray.shape
    hist = {}
    for k in range(nums):
        hist[k] = 0
    for i in range(w):
        for j in range(h):
            if(hist.get(grayArray[i][j]) is None):
                hist[grayArray[i][j]] = 0
            hist[grayArray[i][j]] += 1
    #normalize
    n = w*h
    for key in hist.keys():
        hist[key] = float(hist[key])/n
    return hist


# 计算累计直方图计算出新的均衡化的图片，nums为灰度数,256
def equalization(grayArray, h_s, nums):
    # 计算累计直方图
    tmp = 0.0
    h_acc = h_s.copy()
    for i in range(256):
        tmp += h_s[i]
        h_acc[i] = tmp

    if(len(grayArray.shape) != 2):
        print("length error")
        return None
    w,h = grayArray.shape
    des = np.zeros((w,h),dtype = np.uint8)
    for i in range(w):
        for j in range(h):
            des[i][j] = int((nums - 1)* h_acc[grayArray[i][j] ] +0.5)
    return des


# 传入的直方图要求是个字典，每个灰度对应着概率
def drawHist(hist,name):
    keys = hist.keys()
    values = hist.values()
    x_size = len(hist)-1#x轴长度，也就是灰度级别
    axis_params = []
    axis_params.append(0)
    axis_params.append(x_size)

    #plt.figure()
    if name != None:
        plt.title(name)
    plt.bar(tuple(keys),tuple(values))#绘制直方图
    #plt.show()


# 直方图匹配函数，接受原始图像和目标灰度直方图
def histMatch(grayArray,h_d):
    #计算累计直方图
    tmp = 0.0
    h_acc = h_d.copy()
    for i in range(256):
        tmp += h_d[i]
        h_acc[i] = tmp

    h1 = arrayToHist(grayArray,256)
    tmp = 0.0
    h1_acc = h1.copy()
    for i in range(256):
        tmp += h1[i]
        h1_acc[i] = tmp
    #计算映射
    M = np.zeros(256)
    for i in range(256):
        idx = 0
        minv = 1
        for j in h_acc:
            if (np.fabs(h_acc[j] - h1_acc[i]) < minv):
                minv = np.fabs(h_acc[j] - h1_acc[i])
                idx = int(j)
        M[i] = idx
    des = M[grayArray]
    return des


if __name__ == '__main__':
    imdir = r'D:\database\LEVIR-CD\train\A\train_3.png'
    imdir_match = r'D:\database\LEVIR-CD\train\B\train_3.png'

    #直方图匹配
    #打开文件并灰度化
    im_s = Image.open(imdir).convert("L")
    im_s = np.array(im_s)
    print(np.shape(im_s))
    #打开文件并灰度化
    im_match = Image.open(imdir_match).convert("L")
    im_match = np.array(im_match)
    print(np.shape(im_match))
    #开始绘图
    plt.figure()

    #原始图和直方图
    plt.subplot(2,3,1)
    plt.title("原始图片")
    plt.imshow(im_s,cmap='gray')

    plt.subplot(2,3,4)
    hist_s = arrayToHist(im_s,256)
    drawHist(hist_s,"原始直方图")

    #match图和其直方图
    plt.subplot(2,3,2)
    plt.title("match图片")
    plt.imshow(im_match,cmap='gray')

    plt.subplot(2,3,5)
    hist_m = arrayToHist(im_match,256)
    drawHist(hist_m,"match直方图")

    #match后的图片及其直方图
    im_d = histMatch(im_s,hist_m)#将目标图的直方图用于给原图做均衡，也就实现了match
    plt.subplot(2,3,3)
    plt.title("match后的图片")
    plt.imshow(im_d,cmap='gray')

    plt.subplot(2,3,6)
    hist_d = arrayToHist(im_d,256)
    drawHist(hist_d,"match后的直方图")

    plt.show()