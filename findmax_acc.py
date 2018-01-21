#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 09:15:12 2018

@author: mandren
"""

import numpy as np
from pandas import DataFrame
def findTangent(angle,img):
#这里的angle表示要寻找的切线的方向，img不是原图片，而是截取的只包含物体的轮廓图片
#图片的坐标是从0开始，而不是1
    
    [width,height]=img.size
    img=np.asarray(img)
    a=np.mean(img)
    img.flags.writeable=True
    img[img<=180]=0
    img=img.astype(np.int16)
    if(angle<0):
        angle1=angle+90#angle1表示的是切线的垂直线的角度,angle1一定为正，即slope1一定为正
        slope=np.tan(angle/180*np.pi)#slope一定为负
        slope1=np.tan(angle1/180*np.pi) #slope1表示的是切线的垂直线的斜率
        #x=0;y=0
        #points=[]#这里我们采用的是先把垂直线经过的所有点找出来，然后再在下面对每一个点进行搜索
        
        if(slope1<1):
            num=int(height/0.1)
            y=np.ones((1,num))#here, y.shape=(1,num)
            y=y.cumsum()*0.1# but here, y.shape=(num,)
            x=y/slope1
        else:
            num=int(width/0.1)
            x=np.ones((1,num))
            x=x.cumsum()*0.1
            y=x*slope1
        x=np.round(x);y=np.round(y)
        x=x[x<=width-1];y=y[y<=height-1]
        length=min(len(x),len(y))
        x=list(x);y=list(y)  
        x=x[:length];y=y[:length]      
        p=[x]+[y]
        points=list(zip(*p))
        '''
        while((x<width-1) & (y<height-1)):#这里有减一是因为在下面加一
            if(slope1<1):
                y=y+0.1
                x=np.round(y/slope1)
            else:	
                x=x+0.1
                y=np.round(slope1*x)
            if((np.round(x)<=width-1) & (np.round(y)<=height-1)):#由于在上面中有对x和y进行进一步的更新，所以需要对点重新进行判断
                points.append([x,y])
        '''
        points=np.array(points).astype(np.int16)
        data=DataFrame(data=points,columns=['1','2'])
        data.drop_duplicates(inplace=True)
        points=np.array(data)
        for i in points:#对每一个点进行搜索
            #point=[]
            x=i[0];y=i[1]
            #x1=0;y1=0
            #point.append([x,height-1-y]) 
            if(np.abs(slope)>1):
                num=int(x/0.1)
                x1=np.ones((1,num))*(-1)
                x1=x1.cumsum()*0.1
                y1=x1*slope
            else:
                num=int((height-1-y)/0.1)
                y1=np.ones((1,num))
                y1=y1.cumsum()*0.1
                x1=y1/slope
            x1=np.round(x1);y1=np.round(y1)
            x1=x1[x1>=(-x)];y1=y1[y1<=(height-1-y)]
            x1=x1+x;y1=y1+y;y1=height-1-y1
            length=min(len(x1),len(y1))
            x1=list(x1);y1=list(y1)
            x1=x1[:length];y1=y1[:length]
            p=[x1]+[y1]
            point=list(zip(*p))
              
#下面两个while是因为切线在该点处需要搜索两边的，先搜索左上方的，后搜索右下方的
            '''
            while((x1>(-x)) & (y1<height-1-y)):
                if(np.abs(slope)>1):
                    x1=x1-0.1
                    y1=slope*x1
                else:
                    y1=y1+0.1
                    x1=y1/slope

                if((np.round(x1)>=-x) & (np.round(y1)<=height-1-y)):
                    x11=np.round(x+x1);y11=np.round(y+y1)#变换回以左下角为原点的坐标
                    x111=x11;y111=height-1-y11
                    point.append([x111,y111])
            '''
            if(np.abs(slope)>1):
                num=int((width-1-x)/0.1)
                x1=np.ones((1,num))
                x1=x1.cumsum()*0.1
                y1=x1*slope
            else:
                num=int(y/0.1)
                y1=np.ones((1,num))*(-1)
                y1=y1.cumsum()*0.1
                x1=y1/slope
            x1=np.round(x1);y1=np.round(y1)
            x1=x1[x1<=width-1-x];y1=y1[y1>=(-y)]
            x1=x+x1;y1=y+y1;y1=height-1-y1
            length=min(len(x1),len(y1))
            x1=list(x1);y1=list(y1)
            x1=x1[:length];y1=y1[:length]
            p=[x1]+[y1]
            pointA=list(zip(*p))
            point.extend(pointA)
            point.append((x,height-1-y))
            '''
            x1=0;y1=0
            while((x1<width-1-x) & (y1>-y)):
                if(np.abs(slope)>1):
                    x1=x1+0.1
                    y1=slope*x1
                else:
                    y1=y1-0.1
                    x1=y1/slope
                if((np.round(x1)<=width-1-x) & (np.round(y1)>=-y)):#同样，由于在上面xy有进行迭代，所以要重新检验
    				#这里我们重新变换了坐标，将原点设在每一个点处，所以下面为了能够正确的读取到图片的像素，需要做两轮变换，变换回原来图像的坐标
                    x11=np.round(x+x1);y11=np.round(y+y1)
                    x111=x11;y111=height-1-y11
                    point.append([x111,y111])
            '''
            point=np.array(point).astype(np.int16)
            indexRow=np.array([m for [n,m] in point],dtype=np.int16)
            indexCol=np.array([n for [n,m] in point],dtype=np.int16)
            pointValue=img[indexRow,indexCol]
            if(any(pointValue)):#只要在这条线上有一个点的值不为0.就说明这条线与轮廓相切
                k=np.argmax(pointValue)
                row=indexRow[k]
                col=indexCol[k]
                return [col+2,row-2]
			
		
    else:
        angle1=angle-90
        slope=np.tan(angle/180*np.pi)#slope一定为正
        slope1=np.tan(angle1/180*np.pi)#slope1一定为负
        #x=0;y=0
        #points=[]
        if(np.abs(slope1)<1):
            num=int((height-1)/0.1)
            y=np.ones((1,num))
            y=y.cumsum()*0.1
            x=y/slope1
        else:
            num=int((width-1)/0.1)
            x=np.ones((1,num))*(-1)
            x=x.cumsum()*0.1
            y=slope1*x
        x=np.round(x);y=np.round(y)
        x=x[x>=(1-width)];y=y[y<=height-1]
        length=min(len(x),len(y))
        x=list(x);y=list(y)
        x=x[:length];y=y[:length]
        p=[x]+[y]
        points=list(zip(*p))                
            
        '''
        while((x>1-width) & (y<height-1)):#这里取的坐标是以右下角为原点，所以xy方向按照正常的方向，所以x的坐标是负的	
            if(np.abs(slope1)<1):
                y=y+0.1
                x=round(y/slope1)
            else:
                x=x-0.1
                y=round(slope1*x)
            if((np.round(x)>=1-width) & (np.round(y)<=height-1)):
                points.append([x,y])
        '''
        points=np.array(points).astype(np.int16)
        data=DataFrame(data=points,columns=['1','2'])
        data.drop_duplicates(inplace=True)
        points=np.array(data)
        for i in points:
            [x,y]=i
            #point=[]
            #point.append([width-1+x,height-1-y])
            #x1=0;y1=0
            if(slope>1):
                num=int(abs(x)/0.1)
                x1=np.ones((1,num))
                x1=x1.cumsum()*0.1
                y1=slope*x1
            else:
                num=int((height-1-y)/0.1)
                y1=np.ones((1,num))
                y1=y1.cumsum()*0.1
                x1=y1/slope
            x1=np.round(x1);y1=np.round(y1)
            x1=x1[x1<=abs(x)];y1=y1[y1<=height-1-y]
            x1=x+x1;y1=y+y1;
            x1=width-1+x1;y1=height-1-y1
            x1=list(x1);y1=list(y1)
            length=min(len(x1),len(y1))
            x1=x1[:length];y1=y1[:length]
            p=[x1]+[y1]
            point=list(zip(*p))
            '''两个while，一个搜索右上方的，一个搜索左下方的
            while((x1<abs(x)) & (y1<height-1-y)):
                if(slope>1):
                    x1=x1+0.1
                    y1=slope*x1
                else:
                    y1=y1+0.1
                    x1=y1/slope
            下面需要对坐标重新进行两次变换
                if((np.round(x1)<=abs(x)) & (np.round(y1)<=height-1-y)):
                    x11=np.round(x+x1);y11=np.round(y+y1)
                    x111=width-1+x11;y111=height-1-y11
                    point.append([x111,y111])
            x1=0;y1=0'''
            if(slope1>1):
                num=int((width-1-x)/0.1)
                x1=np.ones((1,num))*(-1)
                x1=x1.cumsum()*0.1
                y1=slope*x1
            else:
                num=int(y/0.1)
                y1=np.ones((1,num))*(-1)
                y1=y1.cumsum()*0.1
                x1=y1/slope
            x1=np.round(x1);y1=np.round(y1)
            x1=x1[x1>=1-x-width];y1=y1[y1>=-y]
            x1=x+x1;y1=y+y1
            x1=width-1+x1;y1=height-1-y1
            x1=list(x1);y1=list(y1)
            length=min(len(x1),len(y1))
            x1=x1[:length];y1=y1[:length]
            p=[x1]+[y1]
            pointA=list(zip(*p))
            point.extend(pointA)
            point.append((width-1+x,height-1-y))
            '''
            while((x1>1-x-width)&(y1>-y)):#x in here is less than 0
                if(slope>1):
                    x1=x1-0.1
                    y1=slope*x1
                else:
                    y1=y1-0.1
                    x1=y1/slope
                if((np.round(x1)>=1-x-width) & (np.round(y1)>=-y)):
                    x11=np.round(x+x1);y11=np.round(y+y1)
                    x111=width-1+x11;y111=height-1-y11
                    point.append([x111,y111])
            '''
            point=np.array(point).astype(np.int16)
            indexRow=np.array([m for [n,m] in point],dtype=np.int16)
            indexCol=np.array([n for [n,m] in point],dtype=np.int16)
            pointValue=img[indexRow,indexCol]
            if(any(pointValue)):#只要在这条线上有一个点的值不为0.就说明这条线与轮廓相切
                k=np.argmax(pointValue)
                row=indexRow[k]
                col=indexCol[k]
                return [col-2,row-2]
def findTangent2(angle,img):
#findTangent1搜索的是图像的下方的切点，而findTangent2搜索的是图像上方的切点
    [width,height]=img.size
    img=np.asarray(img)
    a=np.mean(img)
    img.flags.writeable=True
    img[img<=180]=0
    img=img.astype(np.int16)
    if(angle<0):
        angle1=angle+90#angle1表示的是切线的垂直线的角度,angle1一定为正，即slope1一定为正
        slope=np.tan(angle/180*np.pi)#slope一定为负
        slope1=np.tan(angle1/180*np.pi) #slope1表示的是切线的垂直线的斜率
        #x=0;y=0
        #points=[]
        if(slope1<1):
            num=int((height-1)/0.1)
            y=np.ones((1,num))*(-1)
            y=y.cumsum()*0.1
            x=y/slope1
        else:
            num=int((width-1)/0.1)
            x=np.ones((1,num))*(-1)
            x=x.cumsum()*0.1
            y=slope1*x
        x=np.round(x);y=np.round(y)
        x=x[x>=1-width];y=y[y>=1-height]
        x=list(x);y=list(y)
        length=min(len(x),len(y))
        x=x[:length];y=y[:length]
        p=[x]+[y]
        points=list(zip(*p))
        '''
        while((x>1-width) & (y>1-height)):
            if(slope1<1):
                y=y-0.1
                x=np.round(y/slope1)
            else:	
                x=x-0.1
                y=np.round(slope1*x)
            if((np.round(x)>=1-width) & (np.round(y)>=1-height)):
                points.append([x,y])
        '''
        points=np.array(points).astype(np.int16)
        data=DataFrame(data=points,columns=['1','2'])
        data.drop_duplicates(inplace=True)
        points=np.array(data)
        for i in points:
            #point=[]
            [x,y]=i
            #point.append([width-1+x,-y])
            #x1=0;y1=0
            if(np.abs(slope)>1):
                num=int((width+x-1)/0.1)
                x1=np.ones((1,num))*(-1)
                x1=x1.cumsum()*(0.1)
                y1=slope*x1
            else:
                num=int(-y/0.1)
                y1=np.ones((1,num))
                y1=y1.cumsum()*0.1
                x1=y1/slope
            x1=np.round(x1);y1=np.round(y1)
            x1=x1[x1>=1-x-width];y1=y1[y1<=-y]
            x1=x+x1;y1=y+y1
            x1=width-1+x1;y1=-y1
            x1=list(x1);y1=list(y1)
            length=min(len(x1),len(y1))
            x1=x1[:length];y1=y1[:length]
            p=[x1]+[y1]
            point=list(zip(*p))
            '''
            while((x1>1-x-width) & (y1<-y)):
                if(np.abs(slope)>1):
                    x1=x1-0.1
                    y1=slope*x1
                else:
                    y1=y1+0.1
                    x1=y1/slope
                if((np.round(x1)>=1-x-width) & (np.round(y1)<=-y)):
                    x11=np.round(x+x1);y11=np.round(y+y1)#变换回以左下角为原点的坐标
                    x111=width-1+x11;y111=-y11
                    point.append([x111,y111])
            x1=0;y1=0
            '''
            if(np.abs(slope)>1):
                num=int((-x)/0.1)
                x1=np.ones((1,num))
                x1=x1.cumsum()*0.1
                y1=x1*slope
            else:
                num=int((height+y-1)/0.1)
                y1=np.ones((1,num))*(-1)
                y1=y1.cumsum()*0.1
                x1=y1/slope
            x1=np.round(x1);y1=np.round(y1)
            x1=x1[x1<=-x];y1=y1[y1>=1-y-height]
            x1=x+x1;y1=y+y1
            x1=width-1+x1;y1=-y1
            x1=list(x1);y1=list(y1)
            length=min(len(x1),len(y1))
            x1=x1[:length];y1=y1[:length]
            p=[x1]+[y1]
            pointA=list(zip(*p))
            point.extend(pointA)
            point.append((width-1+x,-y))
            
            '''
            while((x1<-x) & (y1>1-y-height)):
                if(np.abs(slope)>1):
                    x1=x1+0.1
                    y1=slope*x1
                else:
                    y1=y1-0.1
                    x1=y1/slope
                if((np.round(x1)<=-x) & (np.round(y1)>=1-y-height)):
                    x11=np.round(x+x1);y11=np.round(y+y1)
                    x111=width-1+x11;y111=-y11
                    point.append([x111,y111])
            '''
            point=np.array(point).astype(np.int16)
            indexRow=np.array([m for [n,m] in point],dtype=np.int16)
            indexCol=np.array([n for [n,m] in point],dtype=np.int16)
            pointValue=img[indexRow,indexCol]
            if(any(pointValue)):#只要在这条线上有一个点的值不为0.就说明这条线与轮廓相切
                k=np.argmax(pointValue)
                row=indexRow[k]
                col=indexCol[k]
                return [col-2,row+2]
			
		
    else:
        angle1=angle-90
        slope=np.tan(angle/180*np.pi)#slope一定为正
        slope1=np.tan(angle1/180*np.pi)#slope1一定为负
        #x=0;y=0
        #points=[]
        if(np.abs(slope1)<1):
            num=int((height-1)/0.1)
            y=np.ones((1,num))*(-1)
            y=y.cumsum()*0.1
            x=y/slope1
        else:
            num=int((width-1)/0.1)
            x=np.ones((1,num))
            x=x.cumsum()*0.1
            y=slope1*x
        x=np.round(x);y=np.round(y)
        x=x[x<=width-1];y=y[y>=1-height]
        x=list(x);y=list(y)
        length=min(len(x),len(y))
        x=x[:length];y=y[:length]
        p=[x]+[y]
        points=list(zip(*p))
        '''
        while((x<width-1) & (y>1-height)):	
            if(np.abs(slope1)<1):
                y=y-0.1
                x=round(y/slope1)
            else:
                x=x+0.1
                y=round(slope1*x)
            if((np.round(x)<=width-1) & (np.round(y)>=1-height)):
                points.append([x,y])
        '''
        points=np.array(points).astype(np.int16)
        data=DataFrame(data=points,columns=['1','2'])
        data.drop_duplicates(inplace=True)
        points=np.array(data)
        for i in points:
            [x,y]=i
            #point=[]
            #point.append([x,-y])
            #x1=0;y1=0
            if(slope>1):
                num=int((width-x-1)/0.1)
                x1=np.ones((1,num))
                x1=x1.cumsum()*0.1
                y1=slope*x1
            else:
                num=int((-y)/0.1)
                y1=np.ones((1,num))
                y1=y1.cumsum()*0.1
                x1=y1/slope
            x1=np.round(x1);y1=np.round(y1)
            x1=x1[x1<=width-x-1];y1=y1[y1<=-y]
            x1=x+x1;y1=y+y1
            x1=x1;y1=-y1
            x1=list(x1);y1=list(y1)
            length=min(len(x1),len(y1))
            x1=x1[:length];y1=y1[:length]
            p=[x1]+[y1]
            point=list(zip(*p))
            
            '''
            while((x1<width-x-1) & (y1<-y)):
                if(slope>1):
                    x1=x1+0.1
                    y1=slope*x1
                else:
                    y1=y1+0.1
                    x1=y1/slope
    			#下面需要对坐标重新进行两次变换
                if((np.round(x1)<=width-x-1) & (np.round(y1)<=-y)):
                    x11=np.round(x+x1);y11=np.round(y+y1)
                    x111=x11;y111=-y11
                    point.append([x111,y111])
            x1=0;y1=0
            '''
            if(slope>1):
                num=int(x/0.1)
                x1=np.ones((1,num))*(-1)
                x1=x1.cumsum()*0.1
                y1=slope*x1
            else:
                num=int((height+y-1)/0.1)
                y1=np.ones((1,num))*(-1)
                y1=y1.cumsum()*0.1
                x1=y1/slope
            x1=np.round(x1);y1=np.round(y1)
            x1=x1[x1>=-x];y1=y1[y1>=1-height-y]
            x1=x+x1;y1=y+y1
            x1=x1;y1=-y1
            x1=list(x1);y1=list(y1)
            length=min(len(x1),len(y1))
            x1=x1[:length];y1=y1[:length]
            p=[x1]+[y1]
            pointA=list(zip(*p))
            point.extend(pointA)
            point.append((x,-y))
            '''
            while((x1>-x)&(y1>1-height-y)):
                if(slope>1):
                    x1=x1-0.1
                    y1=slope*x1
                else:
                    y1=y1-0.1
                    x1=y1/slope
                if((np.round(x1)>=-x) & (np.round(y1)>=1-height-y)):
                    x11=np.round(x+x1);y11=np.round(y+y1)
                    x111=x11;y111=-y11
                    point.append([x111,y111])
            '''
            point=np.array(point).astype(np.int16)
            indexRow=np.array([m for [n,m] in point],dtype=np.int16)
            indexCol=np.array([n for [n,m] in point],dtype=np.int16)
            pointValue=img[indexRow,indexCol]
            if(any(pointValue)):#只要在这条线上有一个点的值不为0.就说明这条线与轮廓相切
                k=np.argmax(pointValue)
                row=indexRow[k]
                col=indexCol[k]
                return [col+2,row+2]
def findActualScope(img):
    [width,height]=img.size
    img=np.asarray(img)
    a=np.mean(img)
    img.flags.writeable=True
    img[img<=180]=0
    left=0;right=0
    for i in range(width):
        if np.max(img[:,i])>0:
            left=i;break
    for i in range(width):
        if np.max(img[:,width-1-i])>0:
            right=width-1-i;break
    return np.array([left,right])
        
        	
def findIntersection(point1,single1,point2,single2,width,height):
#这个函数计算的是由上面求得切点以及方向之后，求两条直线的交点
    [x1,y1]=point1
    [x2,y2]=point2
    y1=-y1
    y2=-y2 #先转换成正常的坐标系，等算出来之后再转换回来
    slope1=np.tan(single1/180*np.pi)
    slope2=np.tan(single2/180*np.pi)
    c1=y1-slope1*x1
    c2=y2-slope2*x2
    x=(c2-c1)/(slope1-slope2)
    y=(slope1*c2-slope2*c1)/(slope1-slope2)
    x=x if x<width else width-1
    x=x if x>0 else 0
    y=y if -y<height else -height+1 
    y=y if -y>0 else 0
#由于交点可能位于图像外面，所以需要对交点进行进一步的判断，而由2D和3D的图像可以知道，其实3D中的确会有部分点的坐标在2D的外面，所以我们在外面对2D进行截取的时候，需要适当的进行放宽
#需要进行放宽的分别是上下两个边界
    return np.array([int(x),-int(y)])

def findFarPoint(img,point,angle,direction,coord,width):
    img=np.asarray(img)
    [height,width]=img.shape[:2]
    slope=np.tan(angle/180*np.pi)
    [x,y]=point
    y=-y #对y取负，转换为正常的坐标表示
    c=y-x*slope
    if(direction=='left'):
        y1=coord*slope+c
        y1 = y1 if -y1 >=0 else 0
        y1 = y1 if -y1 < height else -height+1
        return([coord,int(-y1)])
    if(direction=='right'):
        k=slope*(coord)+c 
        k=k if -k<height else -height+1
        k=k if -k>=0 else 0
        return(np.array([int(coord),int(-k)]) )
		
def findIntersectionOfVerticalLine(img,point,angle,point2,height):
#这里求的是一条斜线与一条垂直线的交点，则由于tan(90)不存在，所以不能用FindIntersection来进行计算
#point2表示的是垂直线经过的点，point1表示斜线经过的点
    [x1,y1]=point
    y1=-y1 #对y取负，转换为正常的坐标表示
    slope=np.tan(angle/180*np.pi)
    c=y1-x1*slope
    [x2,y2]=point2
    k=slope*x2+c
    k=k if -k<height else -height+1
    k=k if -k>0 else 0
    return np.array([int(x2),-int(k)])
	
	
def Construct3DBox(img,angle1,angle2):
#传入的img是车辆检测得到的轮廓图像，而不是车辆的实际图像，angle1为车辆的运动方向，angle2为道路的方向
    [width,height]=img.size
    [left,right]=findActualScope(img)
    if(angle1<0):
        if((left-8)>0):
            left=left-8
        elif((left-5)>0):
            left=left-5
        elif((left-3)>0):
            left=left-3
        else:
            left=left
        if((right+5)<width):
            right=right+5
        elif((right+5)<width):
            right=right+3
        elif((right+1)<width):
            right=right+1
        else:
            right=right
    else:
        if((left-5)>0):
            left=left-5
        elif((left-3)>0):
            left=left-3
        elif((left-1)>0):
            left=left-1
        else:
            left=left
        if((right+8)<width):
            right=right+8
        elif((right+5)<width):
            right=right+5
        elif((right+3)<width):
            right=right+3
        else:
            right=right
    if(angle1<0):
        [x1,y1]=findTangent(angle1,img)
        [x2,y2]=findTangent(angle2,img)
        point_front_left_down=list(findIntersection([x1,y1],angle1,[x2,y2],angle2,width,height))
        point_rear_left_down=list(findFarPoint(img,[x1,y1],angle1,'left',left,width))
        point_front_right_down=list(findFarPoint(img,[x2,y2],angle2,'right',right,width))
        [x2,y2]=findTangent2(angle1,img)
        point_front_right_up=list(findFarPoint(img,[x2,y2],angle1,'right',right,width)+np.array([0,-4]))
        point_front_left_up=list(findIntersectionOfVerticalLine(img,point_front_right_up,angle2,point_front_left_down,height))
        point_rear_left_up=list(findFarPoint(img,point_front_left_up,angle1,'left',left,width))
        point_rear_right_up=list(findIntersection(point_front_right_up,angle1,point_rear_left_up,angle2,width,height))
    if(angle1>=0):
        [x1,y1]=findTangent(angle1,img)
        [x2,y2]=findTangent(angle2,img)
        point_front_left_down=list(findIntersection([x1,y1],angle1,[x2,y2],angle2,width,height))
        point_front_right_down=list(findFarPoint(img,[x2,y2],angle2,'left',left,width))
        point_rear_left_down=list(findFarPoint(img,[x1,y1],angle1,'right',right,width))
        [x2,y2]=findTangent2(angle1,img)
        point_front_right_up=list(findFarPoint(img,[x2,y2],angle1,'left',left,width)+np.array([0,-4]))
        point_front_left_up=list(findIntersectionOfVerticalLine(img,point_front_right_up,angle2,point_front_left_down,height))
        point_rear_left_up=list(findFarPoint(img,point_front_left_up,angle1,'right',right,width))
        point_rear_right_up=list(findIntersection(point_front_right_up,angle1,point_rear_left_up,angle2,width,height))
    return[point_front_left_down,point_rear_left_down,point_front_right_down,point_front_right_up,point_front_left_up,point_rear_left_up,point_rear_right_up]



