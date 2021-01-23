# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 20:22:46 2021

@author: Mentholatum
"""
import cv2
import numpy as np
import matplotlib as plt

def cv_show(name,imgs):
    cv2.imshow(name,imgs)
    cv2.waitKey() 
    cv2.destroyAllWindows()
#图像轮廓
img1 = cv2.imread(r"G:\Python\Practice\CvProgram\ch01\contours.png",0)
img2 = cv2.imread(r"G:\Python\Practice\CvProgram\ch01\contours2.png",0)
ret,thresh = cv2.threshold(img1,127,255,cv2.THRESH_BINARY)
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
'''传入绘制图像、轮廓索引、颜色模式、线条厚度，
注意需要copy，不然原图会变'''
draw_img = img1.copy()
res = cv2.drawContours(draw_img,contours,-1,(0,0,255),2)
cv_show("result",res)
#轮廓特征
cnt = contours[0]
print(cv2.contourArea(cnt)) 
#轮廓近似
ret, thresh = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnt = contours[0]
draw_img = img2.copy()
res = cv2.drawContours(draw_img, [cnt], -1, (0, 255, 0), 2)
cv_show("result",res)

epsilon = 0.15*cv2.arcLength(cnt,True) #系数表示拟合程度，越小精度越高
approx = cv2.approxPolyDP(cnt,epsilon,True)
draw_img = img2.copy()
res = cv2.drawContours(draw_img, [approx], -1, (255, 0, 0), 2)
cv_show("result",res)
#边界矩形
ret, thresh = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cnt = contours[0]
x,y,w,h = cv2.boundingRect(cnt)
img1 = cv2.rectangle(img1,(x,y),(x+w,y+h),(255,0,0),2)
cv_show("rectangle",img1)








    
    
    
    
    