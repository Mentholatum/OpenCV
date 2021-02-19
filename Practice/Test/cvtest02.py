# -*- coding: utf-8 -*-图像平滑处理
"""
Created on Fri Jan  8 18:34:05 2021

@author: Mentholatum
"""

import cv2
import numpy as np
import matplotlib as plt

def cv_show(name,imgs):
    cv2.imshow(name,imgs)
    cv2.waitKey() 
    cv2.destroyAllWindows()
    
#均值滤波，简单平均的卷积操作
sample = cv2.imread(r"G:\Python\Practice\CvProgram\ch01\lenaNoise.png",1)
cv_show("Original",sample)
blur = cv2.blur(sample,(3,3))
#cv_show("blur",blur)

#方框滤波，和均值基本一样，可以选择归一化，但容易越界
box1 = cv2.boxFilter(sample,-1,(3,3),normalize = True)
box2 = cv2.boxFilter(sample,-1,(3,3),normalize = False)
#cv_show("box1",box1)
#cv_show("box2",box2)

#高斯滤波，卷积矩阵根据高斯函数分配权重
gauss = cv2.GaussianBlur(sample,(5,5),1)
#cv_show("guass",gauss)

#中值滤波，即用中值代替
median = cv2.medianBlur(sample,5)
#cv_show("median",median)

#展示所有结果
res = np.hstack((blur,gauss,median))
cv_show("result",res)

