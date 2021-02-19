# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 21:36:54 2021

@author: Mentholatum
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def cv_show(name,imgs):
    cv2.imshow(name,imgs)
    cv2.waitKey() 
    cv2.destroyAllWindows()
#直方图
img1 = cv2.imread(r"G:\Python\Practice\CvProgram\ch01\cat.jpg",1)
img2 = cv2.imread(r"G:\Python\Practice\CvProgram\ch01\cat.jpg",2)
img3 = cv2.imread(r"G:\Python\Practice\CvProgram\ch01\clahe.jpg",2)
hist = cv2.calcHist([img2],[0],None,[256],[0,256])
#calcHist(images, channels, mask, histSize, ranges[, hist[, accumulate]])最后两个参数一般不变
print(hist.shape)

plt.show()
color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img1],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show() 
#创建mask
mask = np.zeros(img2.shape[:2],np.uint8)#8位无符号整型
mask[100:300,100:400] = 255
masked_img = cv2.bitwise_and(img2,img2,mask = mask)
#cv_show("masked_img",masked_img)
#直方图均衡化
equ = cv2.equalizeHist(img2)
plt.hist(img2.ravel(),256);
plt.hist(equ.ravel(),256);
plt.show()
res = np.hstack((img2,equ))
cv_show("Equalization",res)#即增强对比度
#自适应直方图均衡化
clahe = cv2.createCLAHE(clipLimit = 2.0,tileGridSize = (8,8))
equ = cv2.equalizeHist(img3)
res_clahe = clahe.apply(img3)
res = np.hstack((img3,equ,res_clahe))
cv_show("Clahe Equalization",res)










