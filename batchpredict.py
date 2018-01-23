# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#!/usr/bin/env python
import sys
import subprocess
import os
sys.path.insert(0,pycaffePath)
import caffe
import numpy as np
import pandas as pd
from PIL import Image
import argparse
from findmax_acc import Construct3DBox
from constructFinalImage import transformBoxtoPalne

def transform(filelist,mean_file):
    mean1=np.load(mean_file).mean(1).mean(1)
    data=np.zeros((len(filelist),3,112,112))
    shape=np.zeros((len(filelist),2))
    k=0
    for i in filelist:
        img=Image.open(i)
        im3=np.array(img)
        shape[k,:]=im3.shape[:2]
        im1=img.resize((112,112))
        im1=np.array(im1)
        im2=np.zeros((3,112,112))
        im2[0,:,:]=im1[:,:,0];im2[1,:,:]=im1[:,:,1];im2[2,:,:]=im1[:,:,2]#equal to transformer.set_transpose('data', (2,0,1))
        im3=np.zeros(im2.shape)
        im3[0,:,:]=im2[2,:,:]-mean1[0];im3[1,:,:]=im2[1,:,:]-mean1[1];im3[2,:,:]=im2[0,:,:]-mean1[2]#equal to transformer.set_channel_swap('data', (2,1,0)) & transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
        data[k,:,:,:]=im3
        k=k+1
        img.close()
    return [data,shape]
        
        
    

def compute_real_angle(order,size,num):
    angle=np.zeros((len(order),num))
    for i in range(len(order)):
        order1=order[i][:num]
        size1=size[i]
        for j in range(len(order1)):
            angle1=(order1[j]*3)-90+1.5
            slope1=np.tan(angle1/180*np.pi)#预测的112*112的图片的角
            slope=(size1[0]/size1[1])*slope1#转换成的实际图
            angle2=np.arctan(slope)/np.pi*180
            angle[i,j]=angle2
    return angle
	
def predict_VP2(img_file_list):
    caffe_model=
    deploy_proto=
    net = caffe.Net(deploy_proto,caffe_model,caffe.TEST)
    net.blobs['data'].reshape(len(img_file_list),3,112,112)	
    mean_file=r'/home/mandren/qmk/caffe/VP2-noscale/mean2.npy'
    [data,shape]=transform(img_file_list,mean_file)
    net.blobs['data'].data[...]=data
    out = net.forward()
    prob = net.blobs['prob'].data
    order=(-prob).argsort(axis=1)
    angle1=compute_real_angle(order,shape,2)#这里得到的shape为高，宽，即行数-列数,预测得到的是变化后的图像中的角度，需要转换成原来的图像的角度
    return angle1
def predict_VP1(img_file_list):
    caffe_model=
    deploy_proto=
    net = caffe.Net(deploy_proto,caffe_model,caffe.TEST)
    net.blobs['data'].reshape(len(img_file_list),3,112,112)	
    mean_file=
    [data,shape]=transform(img_file_list,mean_file)
    net.blobs['data'].data[...]=data
    out = net.forward()
    prob = net.blobs['prob'].data
    order=(-prob).argsort(axis=1)
    angle1=compute_real_angle(order,shape,2)#这里得到的shape为高，宽，即行数-列数,预测得到的是变化后的图像中的角度，需要转换成原来的图像的角度
    return angle1
def get2DBoundingBox(img_file_list):
    pwd=os.getcwd()
    os.chdir()
    boxpoint=np.zeros((len(img_file_list),4))
    k=0
    for i in img_file_list:
        subprocess.call(["./darknet", "detect" ,"cfg/yolo.cfg","yolo.weights","%s"%i,"-thresh","0.4"])
        file=open(,'r')
        a=file.readline()
        if(len(a)>0):
            box=tuple(map(int,a.strip().split(',')))
            file.close()
        else:
            file.close()
            subprocess.call(["./darknet", "detect" ,"cfg/yolo.cfg","yolo.weights","%s"%i,"-thresh","0.3"])
            file=open(,'r')
            a=file.readline()
            if(len(a)>0):
                box=tuple(map(int,a.strip().split(',')))
                file.close()
            else:
                file.close()
                subprocess.call(["./darknet", "detect" ,"cfg/yolo.cfg","yolo.weights","%s"%i,"-thresh","0.2"])
                file=open(,'r')
                a=file.readline()
                if(len(a)>0):
                    box=tuple(map(int,a.strip().split(',')))
                    file.close()
                else:
                    file.close()
                    subprocess.call(["./darknet", "detect" ,"cfg/yolo.cfg","yolo.weights","%s"%i,"-thresh","0.1"])
                    file=open(,'r')
                    a=file.readline()
                    box=tuple(map(int,a.strip().split(',')))
                    file.close()
        boxpoint[k]=box
        k=k+1
    os.chdir(pwd)    
    return boxpoint

def preprocessContour(contourimg,box):
    [left,top,right,bottom]=box
    img1=np.array(contourimg)
    img1[:,0:left]=0
    img1[0:top,:]=0
    img1[:,right:]=0
    img1[bottom:,:]=0
    return Image.fromarray(img1)

def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--img_folder",
            help="Input folder where image in"
            )
    parser.add_argument(
            "--contour_file_folder",
            help="Input folder where image contour file in"
            )
    parser.add_argument(
            "--Box_file",
            help="Input file which contain every image's 2D box coord"
            )    
    parser.add_argument(
            "--dst_img_folder",
            help="Input folder where final image store in"
            )
    parser.add_argument(
            "--seperateImg",
            action='store_true',
            help="Determine wheter want to seperate the final image into 3 parts and store them."
            )      
    parser.add_argument(
            "--outputCoord",
            action='store_true',
            help="Determine wheter want to output coord of 7 points or the final image."
            )
    parser.add_argument(
            "--test",
            action='store_true',
            help="Determine wheter want to test the program's performance."
            ) 
    parser.add_argument(
            "--gpu",
            action='store_true',
            help="Switch for gpu computation."
            )
    args = parser.parse_args()
    if args.gpu:
        caffe.set_mode_gpu()
        print("GPU mode")
    else:
        caffe.set_mode_cpu()
        print("CPU mode")
    if(not args.dst_img_folder):
        print("please specify the folder where final image store in")
        return
    if(not(args.img_folder and args.contour_file_folder and args.Box_file )):
        print("You must specify the first three factors")
        return 
    if(not (args.outputCoord or args.test or args.seperateImg)):
        print("please specify at least 1 method you want the program to output the result")
        return
    if args.seperateImg:
        seperateImg=True
    else:
        seperateImg=False
    if args.outputCoord:
        createCoord=True
    else:
        createCoord=False
    if args.test:
        test=True
    else:
        test=False
    dstFolder=args.dst_img_folder
    boxfile=args.Box_file
    point=pd.read_csv(boxfile)
    point.drop(['Unnamed: 0'],axis=1,inplace=True)
    point=np.array(point)
    point[point<0]=0;point=point.astype(np.int16)
    img_file_folder=args.img_folder
    img_file_list=os.listdir(img_file_folder)
    img_file_list.sort()
    for i in range(len(img_file_list)):
        img_file_list[i]=os.path.join(img_file_folder,img_file_list[i])
    
    contour_img_folder=args.contour_file_folder
    contour_img_list=os.listdir(contour_img_folder)
    contour_img_list.sort()
    for i in range(len(img_file_list)):
        contour_img_list[i]=os.path.join(contour_img_folder,contour_img_list[i])
    
    vp1=predict_VP1(img_file_list)
    vp2=predict_VP2(img_file_list)    
#    print(angle1,angle2)
#    boxPoints=get2DBoundingBox(img_file_list)
#    point=tuple(np.array(boxPoints)+np.array([0,-3,0,3]))#这里是对2D bounding box的边界进行扩展，理由见findmax.py中的findIntersection函数
#    print(point)
#    boxImg=contour_img.crop(point)
    length=len(img_file_list)
    dstpoint1=np.zeros((length,7));dstpoint2=np.zeros((length,7))
    if(test):
        if(not os.path.exists(os.path.join(dstFolder,'test'))):
            os.makedirs(os.path.join(dstFolder,'test'))
    if(createCoord):
        if(not os.path.exists(os.path.join(dstFolder,'coord'))):
            os.makedirs(os.path.join(dstFolder,'coord'))
    for i in range(length):
        print(i)
        [vp11,vp12]=vp1[i,:]
        [vp21,vp22]=vp2[i,:]
        if(vp11*vp21<0):
            angle1=vp11;angle2=vp21
        elif(vp11*vp22<0):
            angle1=vp11;angle2=vp22
        elif(vp12*vp21<0):
            angle1=vp12;angle2=vp21
        else:
            angle1=vp12;angle2=vp22
        if(angle1<0): 
            angle1=angle1-5
        else:
            angle1=angle1+5
        contourImg=Image.open(contour_img_list[i])
        contourImg1=preprocessContour(contourImg,point[i])
        points=Construct3DBox(contourImg1,angle1,angle2)        
        if(createCoord):
            p1=[n for [n,m] in points];p2=[m for [n,m] in points]
            dstpoint1[i]=p1;dstpoint2[i]=p2
        if(test):
            for j in points:
                contourImg.putpixel(j,255)
                contourImg.save(os.path.join(dstFolder,'test',str(i)+'.jpg'))
        if(seperateImg):
            img=Image.open(img_file_list[i])
            img1=np.asarray(img)
            img.close()
            [top,left,right]=transformBoxtoPalne(np.array(points),img1,angle1,'seperate')
            os.makedirs(os.path.join(dstFolder,str(i)))
            top.save(os.path.join(dstFolder,str(i),'top.jpg'))
            left.save(os.path.join(dstFolder,str(i),'left.jpg'))
            right.save(os.path.join(dstFolder,str(i),'right.jpg'))
            top.close();right.close();left.close()
        contourImg.close()
        contourImg1.close()
            
    if(createCoord):
        data=pd.DataFrame(data=dstpoint1,columns=['1','2','3','4','5','6','7'])
        data.to_csv(os.path.join(dstFolder,'coord','x.csv'))
        data=pd.DataFrame(data=dstpoint2,columns=['1','2','3','4','5','6','7'])
        data.to_csv(os.path.join(dstFolder,'coord','y.csv'))
	
if __name__ == '__main__':
    main(sys.argv)




