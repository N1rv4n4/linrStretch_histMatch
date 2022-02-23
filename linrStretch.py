import os
import numpy as np
import cv2
from osgeo import gdal, gdal_array
import glob


# 程序不全
def block_max_min_stretch(dataset,filename):
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_bands = dataset.RasterCount
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)
    for i in range(im_bands):
        max_ = np.max(im_data[i, :, :])
        min_ = np.min(im_data[i, :, :])
        for j in range(im_width):
            for k in range(im_height):
                im_data[i,j,k] = (im_data[i,j,k] - min_) / (max_ - min_) * 255

    # driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
    # dataset = driver.Create(filename, im_width, im_height, im_bands, gdal.GDT_Byte)
    #
    # if im_bands == 1:
    #     dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
    # else:
    #     for i in range(im_bands):
    #         dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    #
    del dataset


# 2%裁剪 + 灰度变换
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
    if image.shape[2] == 3:
        b, g, r = cv2.split(image)  # 分开三个波段
        r_p = gray_process(r)
        g_p = gray_process(g)
        b_p = gray_process(b)
        result = cv2.merge((b_p, g_p, r_p))  # 合并处理后的三个波段
        return np.uint8(result)
    if image.shape[2] == 1:
        result = gray_process(image)
        return np.uint8(result)


# 百分比裁剪
def PercentCut(image, max_percent=98, min_percent=2):
    def gray_process(gray):
        high_value = np.percentile(gray, max_percent)  # 取得98%直方图处对应灰度
        print("high_value:", high_value)
        low_value = np.percentile(gray, min_percent)  # 同理
        print("low_value_value:", low_value)
        # np.clip 将灰度值小于low_value的都置为low_value，灰度值大于high_value的都置为high_value
        truncated_gray = np.clip(gray, a_min=low_value, a_max=high_value)
        return truncated_gray
    if image.shape[2] == 3:
        b, g, r = cv2.split(image)  # 分开三个波段
        r_p = gray_process(r)
        g_p = gray_process(g)
        b_p = gray_process(b)
        result = cv2.merge((b_p, g_p, r_p))  # 合并处理后的三个波段
        return np.uint8(result)
    if image.shape[2] == 1:
        result = gray_process(image)
        return np.uint8(result)


# 灰度范围压缩至 1/4
def orign(image):
    b, g, r = cv2.split(image)#分开三个波段
    def gray_process(gray):
        processed_gray = gray/4
        return processed_gray
    r_p = gray_process(r)
    g_p = gray_process(g)
    b_p = gray_process(b)
    result = cv2.merge((b_p, g_p, r_p))#合并处理后的三个波段
    return np.uint8(result)


# 无裁剪，只灰度变换
def max_min_Linear(image, max_out=255, min_out=0):
    b, g, r = cv2.split(image)#分开三个波段

    def gray_process(gray, maxout = max_out, minout = min_out):
        high_value = np.max(gray)
        print("high_value:", high_value)
        low_value = np.min(gray)
        print("low_value:", low_value)
        processed_gray = ((gray - low_value) / (high_value - low_value)) * (maxout - minout)
        return processed_gray
    r_p = gray_process(r)
    g_p = gray_process(g)
    b_p = gray_process(b)
    result = cv2.merge((r_p, g_p, b_p))#合并处理后的三个波段
    return np.uint8(result)


# 255以下置零
def cut_255(image):
    b, g, r = cv2.split(image)
    # 大于255的不变，小于255的为 0
    ret,r_p = cv2.threshold(r, 255, 255, cv2.THRESH_TOZERO)
    ret, g_p = cv2.threshold(g, 255, 255, cv2.THRESH_TOZERO)
    ret, b_p = cv2.threshold(b, 255, 255, cv2.THRESH_TOZERO)

    result = cv2.merge((b_p,g_p,r_p))
    return np.uint8(result)


# 测试
def aaa(image):
    b, g, r = cv2.split(image)
    def gray_process(gray):
        max = np.max(gray)
        min = np.min(gray)
        ave = np.mean(gray)
        # print(ave)
        # var = np.var(gray)
        # print(var)
        # tb = np.max([ave+var*3,max-var*0.5,ave+55])
        # for i in range(512):
        #     for j in range(512):
        #         if gray[i,j]<tb:
        #             gray[i,j] = tb
        #             gray[i, j] = ((gray[i, j] - min) / (tb - min)) * 255
        truncated_gray = gray
        #truncated_gray = np.clip(gray, a_min=min, a_max=max)

        processed_gray = ((truncated_gray - ave) / (max - ave)) * 255
        return processed_gray

    r_p = gray_process(r)
    g_p = gray_process(g)
    b_p = gray_process(b)
    result = cv2.merge((b_p, g_p, r_p))  # 合并处理后的三个波段
    return np.uint8(result)


if __name__ == '__main__':
    # 创建字典 mlist
    mlist={0:'TwoPercent',1:'OnePercent',2:'cut_255',3:'ave',4:'origin'}
    # 加 r 防止转义
    path = r'D:\database\linearStretch'
    save_path = r'D:\database\linearStretch\result'
    # 返回列表，包含所有匹配通配符的tiff文件路径
    tiff_list = glob.glob(path + '/*.tiff')
    # 返回指定的文件夹包含的文件或文件夹的名字的列表
    folder = os.listdir(path)
    save_path2 = save_path + '/' + mlist[1] + '//'
    if not os.path.exists(save_path2):
        os.mkdir(save_path2)

    for idx, tiff in enumerate(tiff_list):
        dataset = gdal.Open(tiff)
        # data = block_max_min_stretch(dataset,filename)
        # 读取第一个tiff图像第一个波段band的数据类型
        datatype = dataset.GetRasterBand(1).DataType
        image = np.zeros((dataset.RasterYSize, dataset.RasterXSize, dataset.RasterCount),
                         dtype=gdal_array.GDALTypeCodeToNumericTypeCode(datatype))
        for b in range(dataset.RasterCount):
            band = dataset.GetRasterBand(b + 1)  # tiff图像的通道数是从 1 开始的
            # ReadAsArray(<xoff>, <yoff>, <xsize>, <ysize>)，读出从(xoff,yoff)开始，大小为(xsize,ysize)的矩阵。
            image[:, :, b] = band.ReadAsArray()

        result = TwoPercentLinear(image)
        # result = max_min_Linear(image)
        # result = cut_255(image)
        # result = aaa(image)
        # result = orign(image)

        filename = save_path2 + os.path.basename(tiff_list[idx])
        cv2.imwrite(filename, result)
