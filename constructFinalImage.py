#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 10:32:13 2018

@author: mandren
"""
import numpy as np
import cv2
from PIL import Image
def transformBoxtoPalne(a,img,angle1,method='seperate'):
	#这里的a表示的是在前面预测好的车辆的3D bounding box的七个点,type(a)=np.array
    if(angle1<0):
        height=abs(a[0]-a[4])[1]
        k=np.abs(a[5]-a[4])
        d=k**2
        width=int(round(np.sqrt(d[0]+d[1])))
        point=np.zeros((4,2))
        point[0]=a[5];point[1]=a[4];point[2]=a[1];point[3]=a[0]
        point=point.astype(np.float32)
        point1=np.array([[0,0],[width,0],[0,height],[width,height]])#这里我们所采用的坐标，仍然和图片所采用的坐标相同
        point1=point1.astype(np.float32)
        Perspectivematrix=cv2.getPerspectiveTransform(point,point1)
        PerspectiveImgLeft = cv2.warpPerspective(img, Perspectivematrix, (width,height))#第三个参数表示我们希望得到的部分的长宽，而在上面我们已经给point赋予了point的坐标，而且是从(0,0)开始的，所以指定width
#和height就只会输出希望得到的部分
        point[0]=a[4];point[1]=a[3];point[2]=a[0];point[3]=a[2]
        k=np.abs(a[3]-a[4])
        d=k**2
        width1=int(round(np.sqrt(d[0]+d[1])))
        point=point.astype(np.float32)
        point1=np.array([[0,0],[width1,0],[0,height],[width1,height]])
        point1=point1.astype(np.float32)
        Perspectivematrix=cv2.getPerspectiveTransform(point,point1)
        PerspectiveImgRight = cv2.warpPerspective(img, Perspectivematrix, (width1,height))
        point[0]=a[6];point[1]=a[3];point[2]=a[5];point[3]=a[4]
        width2=width
        height2=width1
        point=point.astype(np.float32)
        point1=np.array([[0,0],[width2,0],[0,height2],[width2,height2]])
        point1=point1.astype(np.float32)
        Perspectivematrix=cv2.getPerspectiveTransform(point,point1)
        PerspectiveImgTop = cv2.warpPerspective(img, Perspectivematrix, (width2,height2))
        if(method=='whole'):
            FinalImage=np.zeros((height+width1,width+width1,3),dtype=np.uint8)
            FinalImage[0:height2,0:width2,:]=PerspectiveImgTop
            FinalImage[height2:,0:width,:]=PerspectiveImgLeft
            FinalImage[height2:,width:,:]=PerspectiveImgRight
            im=Image.fromarray(FinalImage)
            im=im.transpose(Image.FLIP_LEFT_RIGHT)#在这里我们统一是令最后的图片中左上角为空白，所以在这个角度上需要进行左右翻转，而下面的就不用
            return im
        elif(method=='seperate'):
            PerspectiveImgTop=Image.fromarray(PerspectiveImgTop)
            PerspectiveImgLeft=Image.fromarray(PerspectiveImgLeft)
            PerspectiveImgRight=Image.fromarray(PerspectiveImgRight)
            PerspectiveImgTop=PerspectiveImgTop.transpose(Image.FLIP_LEFT_RIGHT)
            PerspectiveImgLeft=PerspectiveImgLeft.transpose(Image.FLIP_LEFT_RIGHT)
            PerspectiveImgRight=PerspectiveImgRight.transpose(Image.FLIP_LEFT_RIGHT)
            return[PerspectiveImgTop,PerspectiveImgLeft,PerspectiveImgRight]  
        else:
            return 1
    else:
        height=abs(a[0]-a[4])[1]
        k=np.abs(a[5]-a[4])
        d=k**2
        width=int(round(np.sqrt(d[0]+d[1])))
        point=np.zeros((4,2))
        point[0]=a[4];point[1]=a[5];point[2]=a[0];point[3]=a[1]
        point=point.astype(np.float32)
        point1=np.array([[0,0],[width,0],[0,height],[width,height]])
        point1=point1.astype(np.float32)
        Perspectivematrix=cv2.getPerspectiveTransform(point,point1)
        PerspectiveImgLeft = cv2.warpPerspective(img, Perspectivematrix, (width,height))
        point[0]=a[3];point[1]=a[4];point[2]=a[2];point[3]=a[0]
        k=np.abs(a[3]-a[4])
        d=k**2
        width1=int(round(np.sqrt(d[0]+d[1])))
        point=point.astype(np.float32)
        point1=np.array([[0,0],[width1,0],[0,height],[width1,height]])
        point1=point1.astype(np.float32)
        Perspectivematrix=cv2.getPerspectiveTransform(point,point1)
        PerspectiveImgRight = cv2.warpPerspective(img, Perspectivematrix, (width1,height))
        point[0]=a[3];point[1]=a[6];point[2]=a[4];point[3]=a[5]
        width2=width
        height2=width1
        point=point.astype(np.float32)
        point1=np.array([[0,0],[width2,0],[0,height2],[width2,height2]])
        point1=point1.astype(np.float32)
        Perspectivematrix=cv2.getPerspectiveTransform(point,point1)
        PerspectiveImgTop = cv2.warpPerspective(img, Perspectivematrix, (width2,height2))
        if(method=='whole'):
            FinalImage=np.zeros((height+width1,width+width1,3),dtype=np.uint8)
            FinalImage[0:height2,width1:,:]=PerspectiveImgTop
            FinalImage[height2:,width1:,:]=PerspectiveImgLeft
            FinalImage[height2:,0:width1,:]=PerspectiveImgRight
            im=Image.fromarray(FinalImage)
            return im
        elif(method=='seperate'):
            PerspectiveImgTop=Image.fromarray(PerspectiveImgTop)
            PerspectiveImgLeft=Image.fromarray(PerspectiveImgLeft)
            PerspectiveImgRight=Image.fromarray(PerspectiveImgRight)
            return [PerspectiveImgTop,PerspectiveImgLeft,PerspectiveImgRight]

