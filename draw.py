# -*- coding:utf-8 -*-
"""
Created by: 80760
Date: 2022.03.31
"""
# 绘制分段函数

import matplotlib.pyplot as plt
import numpy as np
import math

plt.rcParams['axes.unicode_minus'] = False

x = np.arange(0, 255, 0.1)
print(x)
# y = 45.9891 * np.log(x+1)
# y = x**2/254.8020
# interval0 = [1 if (i < 20) else 0 for i in x]
# interval1 = [1 if (i >= 20 and i < 235) else  for i in x]
# interval2 = [1 if (i >= 235) else 0 for i in x]
#
# y = ((51 * (x - 20)/43) * interval1)
y = np.array([])


def f(x1):
    if x1 < 20:
        return 0
    elif x1 > 235:
        return 255
    else:
        return 51 * (x1 - 20)/43


for v in x:
    y = np.append(y, f(v))
    print(f(v))
    print(y.shape)



print(max(y))
plt.xlabel('r')
plt.ylabel('s', rotation=0)
plt.axis([0, 255, 0, 255])
plt.plot(x, y)
plt.show()
