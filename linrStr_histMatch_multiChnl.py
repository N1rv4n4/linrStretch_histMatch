# -*- coding:utf-8 -*-
"""
Created by: 80760
Date: 2021.12.20
"""
import cv2 as cv
import numpy as np
from histMatch_multiChnl import sigChnlHistMatch
from linrStretch import TwoPercentLinear, PercentCut

if __name__ == '__main__':
    for i in range(24, 25):
        imdir = r"D:\database\LEVIR-CD\train\A\train_{}.png".format(str(i))
        imdir_match = r"D:\database\LEVIR-CD\train\B\train_{}.png".format(str(i))

        im_s = cv.imread(imdir)
        im_match = cv.imread(imdir_match)

        im_s = PercentCut(im_s, max_percent=100, min_percent=0)
        # im_s = TwoPercentLinear(im_s, max_percent=100, min_percent=4)
        cv.imshow('ls', im_s)

        b1, g1, r1 = cv.split(im_s)
        b2, g2, r2 = cv.split(im_match)

        b1m = np.uint8(sigChnlHistMatch(b1, b2))
        g1m = np.uint8(sigChnlHistMatch(g1, g2))
        r1m = np.uint8(sigChnlHistMatch(r1, r2))

        result = cv.merge((b1m, g1m, r1m))
        cv.imshow('rst',result)
        cv.waitKey(0)
        # cv.imwrite("/home/booker/lmk/database/LEVIR-CD/val/AM/val_{}.png".format(str(i)), result)
        print('No.{} histMatch Finished'.format(str(i)))

