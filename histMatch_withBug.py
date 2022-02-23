# -*- coding:utf-8 -*-
"""
Created by: 80760
Date: 2021.11.19
"""
import numpy as np
import cv2 as cv


# 直方图匹配函数，接受原始图像 grayArray 和目标灰度直方图 h_d
def histMatch(grayArray,h_d):
    # 计算目标图像累计直方图
    tmp = 0.0
    h_acc = h_d.copy()
    for i in range(256):
        tmp += h_d[i]
        h_acc[i] = tmp
    # 计算原图累计直方图
    h1 = cv.calcHist([grayArray], [0], None, [256], [0, 256])
    tmp = 0.0
    h1_acc = h1.copy()
    for i in range(256):
        tmp += h1[i]
        h1_acc[i] = tmp
    # 计算映射
    M = np.zeros(256)
    for i in range(256):
        idx = 0
        minv = 1
        for j in range(256):
            if (np.fabs(h_acc[j] - h1_acc[i]) < minv):
                minv = np.fabs(h_acc[j] - h1_acc[i])
                idx = int(j)
        M[i] = idx
    des = M[grayArray]
    return des


orgImg = cv.imread(r'D:\database\LEVIR-CD\train\A\train_1.png')
trgImg = cv.imread(r'D:\database\LEVIR-CD\train\B\train_1.png')
b_o, g_o, r_o = cv.split(orgImg)
b_t, g_t, r_t = cv.split(trgImg)
hist_bt = cv.calcHist([b_t], [0], None, [256], [0, 256])
hist_gt = cv.calcHist([g_t], [0], None, [256], [0, 256])
hist_rt = cv.calcHist([r_t], [0], None, [256], [0, 256])
b_om = histMatch(b_o, hist_bt)
g_om = histMatch(g_o, hist_gt)
r_om = histMatch(r_o, hist_rt)
orgImgM = np.uint8(cv.merge((b_om, g_om, r_om)))
cv.imshow('orgImgM', orgImgM)
cv.waitKey(0)


