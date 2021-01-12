# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 20:29:05 2021

@author: Mentholatum
"""

import cv2
import numpy as np
import matplotlib as plt

def cv_show(name,imgs):
    cv2.imshow(name,imgs)
    cv2.waitKey() 
    cv2.destroyAllWindows()
img = cv2.imread(r"G:\Python\Practice\CvProgram\ch01\dige.png",1)
pie = cv2.imread(r"G:\Python\Practice\CvProgram\ch01\pie.png",1)
lina =cv2.imread(r"G:\Python\Practice\CvProgram\ch01\lena.jpg",2)

#形态学——腐蚀操作
kernel_dige = np.ones((3,3),np.uint8)#卷积矩阵大小为3*3
erosion = cv2.erode(img,kernel_dige,iterations=1)

#iteration表示迭代次数
kernel_pie1 = np.ones((30,30),np.uint8)
erosion_1 = cv2.erode(pie,kernel_pie1,iterations=1)
erosion_2 = cv2.erode(pie,kernel_pie1,iterations=2)
erosion_3 = cv2.erode(pie,kernel_pie1,iterations=3)
res2 = np.hstack((erosion_1,erosion_2,erosion_3))
cv_show("Different iterations",res2)

#形态学——膨胀操作
dige_dilate = cv2.dilate(erosion,kernel_dige,iterations=1)
res1 = np.hstack((img,erosion,dige_dilate))
cv_show("result",res1)

#开运算：先腐蚀再膨胀
opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel_dige)
res3 = np.hstack((img,opening))
cv_show("Opening",res3)

#闭运算：先膨胀再腐蚀
closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel_dige)
res4 = np.hstack((img,closing))
cv_show("Closing",res4)

#梯度运算：膨胀 - 腐蚀
kernel_pie2 = np.ones((7,7),np.uint8)
erosion_4 = cv2.erode(pie,kernel_pie2,iterations=5)
dilate_4 = cv2.dilate(pie,kernel_pie2,iterations=5)
gradient = cv2.morphologyEx(pie,cv2.MORPH_GRADIENT,kernel_pie2)
res5 = np.hstack((erosion_4,dilate_4,gradient))
cv_show("Gradient",res5)

#礼帽：原始输入 - 开运算结果
tophat = cv2.morphologyEx(img,cv2.MORPH_TOPHAT,kernel_dige)
#黑帽：闭运算 - 原始输入
blackhat = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel_dige)
res6 = np.hstack((img,tophat,blackhat))
cv_show("Tophat and Blackhat",res6)

#图像梯度计算——sobel算子（边缘检测）
sobelx = cv2.Sobel(pie,cv2.CV_64F,1,0,ksize = 3)
sobely = cv2.Sobel(pie,cv2.CV_64F,0,1,ksize = 3)
res7 = np.hstack((sobelx,sobely))
cv_show("Sobelx and Sobely",res7)

sobelxy_1 = cv2.Sobel(pie,cv2.CV_64F,1,1,ksize = 3)
sobelxy_2 = cv2.addWeighted(sobelx,0.5,sobely,0.5,0)#偏置项默认为0
res8 = np.hstack((sobelxy_1,sobelxy_2))
cv_show("Sobelxy and addWeight",res8)

#当图像边界较为复杂时，需要添加绝对值函数
sobelx_lina = cv2.Sobel(lina,cv2.CV_64F,1,0,ksize = 3)
sobelx_lina = cv2.convertScaleAbs(sobelx_lina)#必不可少
sobely_lina = cv2.Sobel(lina,cv2.CV_64F,0,1,ksize = 3)
sobely_lina = cv2.convertScaleAbs(sobely_lina)
sobelxy_lina = cv2.addWeighted(sobelx_lina,0.5,sobely_lina,0.5,0)



#图像梯度计算——Scharr算子，对边界更敏感
scharrx_lina = cv2.Scharr(lina,cv2.CV_64F,1,0)#无需指定卷积矩阵大小
scharrx_lina = cv2.convertScaleAbs(scharrx_lina)
scharry_lina = cv2.Scharr(lina,cv2.CV_64F,0,1)
scharry_lina = cv2.convertScaleAbs(scharry_lina)
scharrxy_lina = cv2.addWeighted(scharrx_lina,0.5,scharry_lina,0.5,0)
#图像梯度计算——laplacian算子（二阶导数），对噪音点比较敏感
laplacian_lina = cv2.Laplacian(lina,cv2.CV_64F)#不区分xy方向
laplacian_lina = cv2.convertScaleAbs(laplacian_lina)
res9 = np.hstack((lina,sobelxy_lina,scharry_lina,laplacian_lina))
cv_show("Lena, Sobel, Scahrr, Laplacian",res9)
