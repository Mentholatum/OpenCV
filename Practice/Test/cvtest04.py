# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 18:09:26 2021

@author: Mentholatum
"""
import cv2
import numpy as np
import matplotlib as plt

def cv_show(name,imgs):
    cv2.imshow(name,imgs)
    cv2.waitKey() 
    cv2.destroyAllWindows()
#Canny边缘检测
lina = cv2.imread(r"G:\Python\Practice\CvProgram\ch01\lena.jpg",2)
car = cv2.imread(r"G:\Python\Practice\CvProgram\ch01\car.png",2)
v1 = cv2.Canny(lina,80,150)
v2 = cv2.Canny(lina,50,100)
res1 = np.hstack((lina,v1,v2))
cv_show("result",res1)

v1 = cv2.Canny(car,120,250)
v2 = cv2.Canny(car,50,100)
res1 = np.hstack((v1,v2))
cv_show("result1(120,250)   result2(50,100)",res1)

#图像金字塔 向上采样（放大） 向下采样（缩小）
#高斯金字塔
am = cv2.imread(r"G:\Python\Practice\CvProgram\ch01\am.png",1)
print(am.shape)
up = cv2.pyrUp(am)
print(up.shape)
down = cv2.pyrDown(am)
print(down.shape)
cv_show("original am",am)
cv_show("up am",up)
cv_show("down am",down)
#拉普拉斯金字塔——1、低通滤波 2、缩小尺寸 3、放大尺寸 4、图像相减
down_up = cv2.pyrUp(down)
l_1 = am - down_up
cv_show("laplas",l_1)






