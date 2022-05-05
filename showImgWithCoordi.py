# -*- coding:utf-8 -*-
"""
Created by: 80760
Date: 2022.04.14
"""

import cv2
import sys
import csv

#鼠标事件
def mouse(event, x, y, flags, param):
    global flag, horizontal, vertical, flag_hor, flag_ver, dx, dy, sx, sy, dst, x1, y1, x2, y2, x3, y3, f1, f2
    global zoom, scroll_har, scroll_var, img_w, img_h, img, dst1, win_w, win_h, show_w, show_h, CoordinateX, CoordinateY
    if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
        if flag == 0:
            if horizontal and 0 < x < win_w and win_h - scroll_w < y < win_h:
                flag_hor = 1  # 鼠标在水平滚动条上
            elif vertical and win_w - scroll_w < x < win_w and 0 < y < win_h:
                flag_ver = 1  # 鼠标在垂直滚动条上
            else:  #鼠标在图像上
                truex = x + dx
                truey = y + dy
                truexy = "%d,%d" % (truex, truey)
                CoordinateX.append(truex)
                CoordinateY.append(truey)
                img1 = img[dy:dy + show_h, dx:dx + show_w]  # 截取显示图片
                cv2.circle(img1, (x, y), 1, (255, 255, 255), thickness=-1)
                cv2.putText(img1, truexy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                            1.0, (255, 255, 255), thickness=1)
                dst = img1.copy()

            if flag_hor or flag_ver:
                flag = 1  # 进行滚动条垂直
                x1, y1, x2, y2, x3, y3 = x, y, dx, dy, sx, sy  # 使鼠标移动距离都是相对于初始滚动条点击位置，而不是相对于上一位置
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
        if flag == 1:
            if flag_hor:
                w = (x - x1)/2  # 移动宽度
                dx = x2 + w * f1  # 原图x
                if dx < 0:  # 位置矫正
                    dx = 0
                elif dx > img_w - show_w:
                    dx = img_w - show_w
                sx = x3 + w  # 滚动条x
                if sx < 0:  # 位置矫正
                    sx = 0
                elif sx > win_w - scroll_har:
                    sx = win_w - scroll_har
            if flag_ver:
                h = y - y1  # 移动高度
                dy = y2 + h * f2  # 原图y
                if dy < 0:  # 位置矫正
                    dy = 0
                elif dy > img_h - show_h:
                    dy = img_h - show_h
                sy = y3 + h  # 滚动条y
                if sy < 0: # 位置矫正
                    sy = 0
                elif sy > win_h - scroll_var:
                    sy = win_h - scroll_var
            dx, dy = int(dx), int(dy)
            img1 = img[dy:dy + show_h, dx:dx + show_w]  # 截取显示图片
            dst = img1.copy()
    elif event == cv2.EVENT_LBUTTONUP:  # 左键释放
        flag, flag_hor, flag_ver = 0, 0, 0
        x1, y1, x2, y2, x3, y3 = 0, 0, 0, 0, 0, 0

    if horizontal and vertical:
        sx, sy = int(sx), int(sy)
        # 对dst1画图而非dst，避免鼠标事件不断刷新使显示图片不断进行填充
        dst1 = cv2.copyMakeBorder(dst, 0, scroll_w, 0, scroll_w, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        cv2.rectangle(dst1, (sx, show_h), (int(sx + scroll_har), win_h), (181, 181, 181), -1)  # 画水平滚动条
        cv2.rectangle(dst1, (show_w, sy), (win_w, int(sy + scroll_var)), (181, 181, 181), -1)  # 画垂直滚动条
    elif horizontal == 0 and vertical:
        sx, sy = int(sx), int(sy)
        dst1 = cv2.copyMakeBorder(dst, 0, 0, 0, scroll_w, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        cv2.rectangle(dst1, (show_w, sy), (win_w, int(sy + scroll_var)), (181, 181, 181), -1)  # 画垂直滚动条
    elif horizontal and vertical == 0:
        sx, sy = int(sx), int(sy)
        dst1 = cv2.copyMakeBorder(dst, 0, scroll_w, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        cv2.rectangle(dst1, (sx, show_h), (int(sx + scroll_har), win_h), (181, 181, 181), -1)  # 画水平滚动条
    cv2.imshow("img", dst1)
    cv2.waitKey(1)


img = cv2.imread(r"F:\jdata\472_cut/123.tif")  # 此处需换成大于img_w * img_h的图片
cv2.namedWindow('img')
img_h, img_w = img.shape[0:2]  # 原图宽高
show_h, show_w = 600, 800  # 显示图片宽高
horizontal, vertical = 0, 0  # 原图是否超出显示图片
dx, dy = 0, 0  # 显示图片相对于原图的坐标
scroll_w = 16  # 滚动条宽度
sx, sy = 0, 0  # 滚动块相对于滚动条的坐标
flag, flag_hor, flag_ver = 0, 0, 0  # 鼠标操作类型，鼠标是否在水平滚动条上，鼠标是否在垂直滚动条上
x1, y1, x2, y2, x3, y3 = 0, 0, 0, 0, 0, 0  # 中间变量
win_w, win_h = show_w + scroll_w, show_h + scroll_w  # 窗口宽高
scroll_har, scroll_var = win_w * show_w / img_w, win_h * show_h / img_h  # 滚动条水平垂直长度
wheel_step, zoom = 0.05, 1  # 缩放系数， 缩放值
zoom_w, zoom_h = img_w, img_h  # 缩放图宽高
f1, f2 = (img_w - show_w) / (win_w - scroll_har), (img_h - show_h) / (win_h - scroll_var)  # 原图可移动部分占滚动条可移动部分的比例
CoordinateX = []  # 选中点的X坐标集合
CoordinateY = []  # 选中点的Y坐标集合
if img_h <= show_h and img_w <= show_w:
    cv2.imshow("img", img)
    cv2.destroyAllWindows()
    print("1")
    sys.exit(0)
else:
    if img_w > show_w:
        horizontal = 1
    if img_h > show_h:
        vertical = 1
    i = img[dy:dy + show_h, dx:dx + show_w]
    dst = i.copy()

cv2.resizeWindow("img", win_w, win_h)
cv2.setMouseCallback('img', mouse)

cv2.waitKey()
cv2.destroyAllWindows()

print(CoordinateX)
print(CoordinateY)

# print("如需将采样点写入txt请输入1,否则输入0")
# isUpdate = input("input = ")
# while(True):
#     if(isUpdate == "1"):
#         cv2.imwrite("2.jpg",img)
#         csvFile = open("POI.csv", "w", encoding='utf8', newline='')  # 创建csv文件
#         writer = csv.writer(csvFile)  # 创建写的对象
#         for i in range(len(CoordinateX)):
#             writer.writerow([CoordinateX[i],CoordinateY[i]])
#         csvFile.close()
#         break
#     elif(isUpdate == "0"):
#         break
#     else:
#         print("请输入1或0")
#         print("如需采样请输入1,如已采样完毕请输入0")
#         isUpdate = input("input = ")

