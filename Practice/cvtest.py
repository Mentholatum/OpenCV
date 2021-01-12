# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 12:03:02 2021

@author: Mentholatum
"""
import cv2 #opencv读取的默认格式是BGR
import matplotlib.pyplot as plt
img1 = cv2.imread(r"G:\Python\Practice\CvProgram\ch01\cat.jpg",1) #彩色图
img2 = cv2.imread(r"G:\Python\Practice\CvProgram\ch01\cat.jpg",2) #灰色图
'''print(img1)
cv2.imshow("Cat",img)
cv2.waitKey(0) #0表示任意键终止
cv2.destroyAllWindows()'''
def cv_show(name,imgs):
    cv2.imshow(name,imgs)
    cv2.waitKey() 
    cv2.destroyAllWindows()
#cv_show("Cat",img)
'''
print(img1.shape) #h w c height weight dimension
print(img2)'''

#视频读取
vc = cv2.VideoCapture(r"G:\Python\Practice\CvProgram\ch01\test.mp4")
if vc.isOpened():
    open,frame = vc.read()
else:
    open = False
'''
while open:
    ret,frame = vc.read()
    if frame is None:
        break
    if ret == True:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('result',gray)
        if cv2.waitKey(10) & 0xFF == 27:
            break
vc.release()
cv2.destroyAllWindows
'''
#截取部分图片像素
cat = img1[0:160,0:300]
#cv_show("Cut cat",cat)

#显示单通道颜色图片 B0 G1 R2
b,g,r = cv2.split(img1)
R_img = img1.copy()
R_img[:,:,0] = 0
R_img[:,:,1] = 0
#cv_show("RedCat",R_img)

#边界填充
top_size,bottom_size,left_size,right_size = (50,50,50,50)
replicate = cv2.copyMakeBorder(img1, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img1, top_size, bottom_size, left_size, right_size,cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img1, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img1, top_size, bottom_size, left_size, right_size, cv2.BORDER_WRAP)
constant = cv2.copyMakeBorder(img1, top_size, bottom_size, left_size, right_size,cv2.BORDER_CONSTANT, value=0)
'''
plt.subplot(231), plt.imshow(img1, 'gray'), plt.title('ORIGINAL')
plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('REPLICATE')
plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('REFLECT')
plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('REFLECT_101')
plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('WRAP')
plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('CONSTANT')
plt.show()
'''
'''
BORDER_REPLICATE：复制法，也就是复制最边缘像素。
BORDER_REFLECT：反射法，对感兴趣的图像中的像素在两边进行复制例如：fedcba|abcdefgh|hgfedcb
BORDER_REFLECT_101：反射法，也就是以最边缘像素为轴，对称，gfedcb|abcdefgh|gfedcba
BORDER_WRAP：外包装法cdefgh|abcdefgh|abcdefg
BORDER_CONSTANT：常量法，常数值填充。
'''

#数值计算
img_plus1 = img1 + 10
cv_show("Plus",img_plus1)
img_plus2 = cv2.add(img1,img1)
cv_show("Plus Function",img_plus2)

#图像融合 shape值必须相同
img_cat = cv2.imread(r"G:\Python\Practice\CvProgram\ch01\cat.jpg",1)
img_dog = cv2.imread(r"G:\Python\Practice\CvProgram\ch01\dog.jpg",1)
img_dog = cv2.resize(img_dog,(500,414))
img_union = cv2.addWeighted(img_cat,0.5,img_dog,0.5,1)
plt.imshow(img_union)
