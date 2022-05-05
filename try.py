# -*- coding:utf-8 -*-
"""
Created by: 80760
Date: 2021.11.19
"""
import numpy as np
a = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])
b = np.array([5, 6])
# # cv.imshow('1', b)
# # cv.waitKey(0)
# print(a)
print(b[0], b[1])
c = b[a]
print(c)
# print(c.shape)