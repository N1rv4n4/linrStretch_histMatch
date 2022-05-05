# -*- coding:utf-8 -*-
"""
Created by: 80760
Date: 2022.04.26
"""
import os
import shutil

def getOriName(nameWithForm):
    # 去掉文件名中的点及后缀
    str1 = nameWithForm
    str2 = '.'
    dotIndex = str1.index(str2)
    oriName = nameWithForm[:dotIndex]
    return oriName

def getOriNameList(nameListWithForm):

    OriNameList = []
    for fileName in nameListWithForm:
        fileOriName = getOriName(fileName)
        OriNameList.append(fileOriName)
    return OriNameList


if __name__ == '__main__':
    imgPath = r'F:\shipDetecData4\final_real_dataset_onlypositivesamles\clear_images_cut'
    labelPath = r'F:\shipDetecData4\final_real_dataset_onlypositivesamles\Cut_GT'
    blankXMLPath = r'F:\shipDetecData4\final_real_dataset_onlypositivesamles\Cut_GT\4768_6657_2.xml'
    labelSavePath = r'F:\shipDetecData4\final_real_dataset_onlypositivesamles\Cut_GT_Blank'

    imgList = os.listdir(imgPath)
    labelList = os.listdir(labelPath)

    imgOriNameList = getOriNameList(imgList)
    labelOriNameList = getOriNameList(labelList)

    for imgOriName in imgOriNameList:
        if (imgOriName in labelOriNameList) is False:
            shutil.copy(blankXMLPath, labelSavePath + '/' + imgOriName + '.xml')

