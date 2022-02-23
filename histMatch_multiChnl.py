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
import math
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文


# 将灰度数组映射为直方图字典,nums表示灰度的数量级
def arrayToHist(grayArray, nums):
    if (len(grayArray.shape) != 2):
        print("length error")
        return None
    w, h = grayArray.shape
    hist = {}
    for k in range(nums):
        hist[k] = 0
    for i in range(w):
        for j in range(h):
            if (hist.get(grayArray[i][j]) is None):
                hist[grayArray[i][j]] = 0
            hist[grayArray[i][j]] += 1
    # normalize
    n = w * h
    for key in hist.keys():
        hist[key] = float(hist[key]) / n
    return hist


# 计算累计直方图计算出新的均衡化的图片，nums为灰度数,256
# def equalization(grayArray, h_s, nums):
#     # 计算累计直方图
#     tmp = 0.0
#     h_acc = h_s.copy()
#     for i in range(256):
#         tmp += h_s[i]
#         h_acc[i] = tmp
#
#     if(len(grayArray.shape) != 2):
#         print("length error")
#         return None
#     w,h = grayArray.shape
#     des = np.zeros((w,h),dtype = np.uint8)
#     for i in range(w):
#         for j in range(h):
#             des[i][j] = int((nums - 1)* h_acc[grayArray[i][j] ] +0.5)
#     return des


# 传入的直方图要求是个字典，每个灰度对应着概率
def drawHist(hist, name):
    keys = hist.keys()
    values = hist.values()
    x_size = len(hist) - 1  # x轴长度，也就是灰度级别
    axis_params = []
    axis_params.append(0)
    axis_params.append(x_size)

    # plt.figure()
    if name != None:
        plt.title(name)
    plt.bar(tuple(keys), tuple(values))  # 绘制直方图
    # plt.show()


# 直方图匹配函数，接受原始图像和目标灰度直方图
def histMatch(grayArray, h_d):
    # 计算累计直方图
    tmp = 0.0
    h_acc = h_d.copy()
    for i in range(256):
        tmp += h_d[i]
        h_acc[i] = tmp  # h_acc目标积累直方图

    h1 = arrayToHist(grayArray, 256)  # h1 原图直方图
    tmp = 0.0
    h1_acc = h1.copy()
    for i in range(256):
        tmp += h1[i]
        h1_acc[i] = tmp  # h1_acc原图积累直方图
    # 计算映射
    M = np.zeros(256)  # 1*256数组
    for i  in range(256):
        idx = 0
        minv = 1
        for j in h_acc:
            if (np.fabs(h_acc[j] - h1_acc[i]) < minv):  # 对于原图积累直方图的每个灰度，将其与目标积累直方图的各个灰度比较
                minv = np.fabs(h_acc[j] - h1_acc[i])
                idx = int(j)
        M[i] = idx
    # print(M)
    # print(grayArray)
    des = M[grayArray]
    # print(des)
    return des


# 对单通道匹配
def sigChnlHistMatch(scImg, scImg_d):
    im_s = np.array(scImg)
    im_match = np.array(scImg_d)
    hist_m = arrayToHist(im_match, 256)
    im_d = histMatch(im_s, hist_m)
    return im_d


def TwoPercentLinear(image, max_out=255, min_out=0, max_percent=98, min_percent=2):
    def gray_process(gray, maxout = max_out, minout = min_out):
        high_value = np.percentile(gray, max_percent)  # 取得98%直方图处对应灰度
        print("high_value:", high_value)
        low_value = np.percentile(gray, min_percent)  # 同理
        print("low_value_value:", low_value)
        # np.clip 将灰度值小于low_value的都置为low_value，灰度值大于high_value的都置为high_value
        truncated_gray = np.clip(gray, a_min=low_value, a_max=high_value)
        # 将范围 low_value ~ high_value 变换至 0 ~ 255
        processed_gray = ((truncated_gray - low_value) / (high_value - low_value)) * (maxout - minout)
        return processed_gray
    image = gray_process(image)
    return np.uint8(image)


def PercentCut(image, max_percent=98, min_percent=0):
    def gray_process(gray):
        high_value = np.percentile(gray, max_percent)  # 取得98%直方图处对应灰度
        print("max:", np.max(gray), "high_value:", high_value)
        low_value = np.percentile(gray, min_percent)  # 同理
        print("min", np.min(gray), "low_value:", low_value)
        # np.clip 将灰度值小于low_value的都置为low_value，灰度值大于high_value的都置为high_value
        truncated_gray = np.clip(gray, a_min=low_value, a_max=high_value)
        return truncated_gray
    image = gray_process(image)
    return np.uint8(image)


def BestPercent(image):
    maxnum = np.sum(image==255)
    print("maxnum:", maxnum)
    result = 100-math.ceil(100*maxnum/np.size(image))
    print("bestpercent:", result)
    return result


if __name__ == '__main__':
    for i in range(24, 25):
        imdir = r"D:\database\LEVIR-CD\train\A\train_{}.png".format(str(i))
        imdir_match = r"D:\database\LEVIR-CD\train\B\train_{}.png".format(str(i))

        im_s = cv.imread(imdir)
        im_match = cv.imread(imdir_match)
        b1, g1, r1 = cv.split(im_s)
        b2, g2, r2 = cv.split(im_match)

        # 先匹配
        b1m0 = np.uint8(sigChnlHistMatch(b1, b2))
        g1m0 = np.uint8(sigChnlHistMatch(g1, g2))
        r1m0 = np.uint8(sigChnlHistMatch(r1, r2))

        # 自适应
        bp = BestPercent(b1m0)
        gp = BestPercent(g1m0)
        rp = BestPercent(r1m0)
        bestpct = min(bp, gp, rp)

        # 再裁剪
        b1m = PercentCut(b1m0, max_percent=bp)
        g1m = PercentCut(g1m0, max_percent=gp)
        r1m = PercentCut(r1m0, max_percent=rp)

        # # 求直方图
        # b1h = arrayToHist(b1, 256)  # 原图
        # g1h = arrayToHist(g1, 256)
        # r1h = arrayToHist(r1, 256)
        #
        # b2h = arrayToHist(b2, 256)  # 目标图
        # g2h = arrayToHist(g2, 256)
        # r2h = arrayToHist(r2, 256)
        #
        # b1m0h = arrayToHist(b1m0, 256)  # 未裁剪
        # g1m0h = arrayToHist(g1m0, 256)
        # r1m0h = arrayToHist(r1m0, 256)

        result = cv.merge((b1m, g1m, r1m))
        # cv.imwrite(r"D:\database\LEVIR-CD\train\AMC\train_{}.png".format(str(i)), result)
        # cv.imwrite(r".\g1m0.png", g1m0)
        # cv.imwrite(r".\r1m0.png", r1m0)
        print('No.{} histMatch Finished'.format(str(i)), bestpct)

        # plt.figure()
        #
        # plt.subplot(3, 3, 1)
        # plt.title('b1')
        # plt.imshow(b1, cmap=plt.cm.gray)
        #
        # plt.subplot(3, 3, 2)
        # plt.title('g1')
        # plt.imshow(g1, cmap=plt.cm.gray)
        #
        # plt.subplot(3, 3, 3)
        # plt.title('r1')
        # plt.imshow(r1, cmap=plt.cm.gray)
        #
        # plt.subplot(3, 3, 4)
        # plt.title('b1m0')
        # plt.imshow(b1m0, cmap=plt.cm.gray)
        #
        # plt.subplot(3, 3, 5)
        # plt.title('g1m0')
        # plt.imshow(g1m0, cmap=plt.cm.gray)
        #
        # plt.subplot(3, 3, 6)
        # plt.title('r1m0')
        # plt.imshow(r1m0, cmap=plt.cm.gray)
        #
        # plt.subplot(3, 3, 7)
        # plt.title('b1m')
        # plt.imshow(b1m, cmap=plt.cm.gray)
        #
        # plt.subplot(3, 3, 8)
        # plt.title('g1m')
        # plt.imshow(g1m, cmap=plt.cm.gray)
        #
        # plt.subplot(3, 3, 9)
        # plt.title('r1m')
        # plt.imshow(r1m, cmap=plt.cm.gray)
        #
        # plt.tight_layout()
        # plt.show()

        # plt.savefig('./results.png')
        # cv.imshow('1', result)
        # cv.waitKey(0)
