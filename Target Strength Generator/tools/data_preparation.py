import pandas as pd
from math import dist
import statistics
import math
import os
import xml.etree.ElementTree as ET

#walleye data part
def read_walleye(walleye_data,path):
    walleye_list =[]
    for picture in walleye_data['frames']:
        k=len(picture['labels'])
        for item in picture['labels']:
            if(k==1):
                a=(item['box2d']['x1'],item['box2d']['y1'])#pixel
                b=(item['box2d']['x2'],item['box2d']['y2'])
                short=item['box2d']['x2']-item['box2d']['x1']
                w=item['box2d']['x2']-item['box2d']['x1']
                h=item['box2d']['y2']-item['box2d']['y1']
                long=dist(a,b)
                TS=(20.96*(math.log10((long)*1500/(637.16*2))))-97.05+20.96
                locx=statistics.mean([item['box2d']['x1'],item['box2d']['x2']]) #bounding box中心點X
                locy=statistics.mean([item['box2d']['y1'],item['box2d']['y2']])#bounding box中心點y, 坐標軸原點在左上角
                distance=dist((locx,locy),(200,0))
                walleye_list.append([item['id'],item['box2d']['x1'],item['box2d']['x2'],item['box2d']['y1'],item['box2d']['y2'],item['box2d']['x2']-item['box2d']['x1'],
                            dist(a,b),locx,locy,path+'/DIDSON_dataset/walleye_label_5028'+picture['name'][42:],TS,item['box2d']['x1'],item['box2d']['y1'],w,h,distance])


    df=pd.DataFrame(walleye_list,columns=['ID','x1','x2','y1','y2','minlength','Maxlength','location_x','location_y','file','TS','x','y','w','h','distance'])
    df1=df.sort_values(by=['file'])
    return df1

# smbass data part
def read_smbass(smbass_data,path):
    smbass_list =[]
    for picture in smbass_data['frames']:
        k=len(picture['labels'])
        for item in picture['labels']:
            if(k==1):
                a=(item['box2d']['x1'],item['box2d']['y1'])#pixel
                b=(item['box2d']['x2'],item['box2d']['y2'])
                short=item['box2d']['x2']-item['box2d']['x1']
                w=item['box2d']['x2']-item['box2d']['x1']
                h=item['box2d']['y2']-item['box2d']['y1']
                long=dist(a,b)
                TS=(20.96*(math.log10((long)*1500/(637.16*2))))-97.05+20.96
                locx=statistics.mean([item['box2d']['x1'],item['box2d']['x2']]) #bounding box中心點X
                locy=statistics.mean([item['box2d']['y1'],item['box2d']['y2']])#bounding box中心點y, 坐標軸原點在左上角
                distance=dist((locx,locy),(200,0))
                smbass_list.append([item['id'],item['box2d']['x1'],item['box2d']['x2'],item['box2d']['y1'],item['box2d']['y2'],item['box2d']['x2']-item['box2d']['x1'],
                            dist(a,b),locx,locy,path+'/DIDSON_dataset/smbass_label_5040'+picture['name'][41:],TS,item['box2d']['x1'],item['box2d']['y1'],w,h,distance])


    df=pd.DataFrame(smbass_list,columns=['ID','x1','x2','y1','y2','minlength','Maxlength','location_x','location_y','file','TS','x','y','w','h','distance'])
    df2=df.sort_values(by=['file'])
    return df2

# carp data part
def read_carp(carp_data,path):
    carp_list =[]
    for picture in carp_data['frames']:
        k=len(picture['labels'])
        for item in picture['labels']:
            if(k==1):
                a=(item['box2d']['x1'],item['box2d']['y1'])#pixel
                b=(item['box2d']['x2'],item['box2d']['y2'])
                short=item['box2d']['x2']-item['box2d']['x1']
                w=item['box2d']['x2']-item['box2d']['x1']
                h=item['box2d']['y2']-item['box2d']['y1']
                long=dist(a,b)
                TS=(20.96*(math.log10((long)*1500/(637.16*2))))-97.05+20.96
                locx=statistics.mean([item['box2d']['x1'],item['box2d']['x2']]) #bounding box中心點X
                locy=statistics.mean([item['box2d']['y1'],item['box2d']['y2']])#bounding box中心點y, 坐標軸原點在左上角
                distance=dist((locx,locy),(200,0))
                carp_list.append([item['id'],item['box2d']['x1'],item['box2d']['x2'],item['box2d']['y1'],item['box2d']['y2'],item['box2d']['x2']-item['box2d']['x1'],
                            dist(a,b),locx,locy,path+'/DIDSON_dataset/carp_label_5095'+picture['name'][39:],TS,item['box2d']['x1'],item['box2d']['y1'],w,h,distance])


    df=pd.DataFrame(carp_list,columns=['ID','x1','x2','y1','y2','minlength','Maxlength','location_x','location_y','file','TS','x','y','w','h','distance'])
    df3=df.sort_values(by=['file'])
    return df3

# lamprey data part
def read_lamprey(lamprey_data,path):
    lamprey_list =[]
    for picture in lamprey_data['frames']:
        k=len(picture['labels'])
        for item in picture['labels']:
            if(k==1):
                a=(item['box2d']['x1'],item['box2d']['y1'])#pixel
                b=(item['box2d']['x2'],item['box2d']['y2'])
                short=item['box2d']['x2']-item['box2d']['x1']
                w=item['box2d']['x2']-item['box2d']['x1']
                h=item['box2d']['y2']-item['box2d']['y1']
                long=dist(a,b)
                locx=statistics.mean([item['box2d']['x1'],item['box2d']['x2']]) #bounding box中心點X
                locy=statistics.mean([item['box2d']['y1'],item['box2d']['y2']])#bounding box中心點y, 坐標軸原點在左上角
                distance=dist((locx,locy),(200,0))
                lamprey_list.append([item['id'],item['box2d']['x1'],item['box2d']['x2'],item['box2d']['y1'],item['box2d']['y2'],item['box2d']['x2']-item['box2d']['x1'],
                            dist(a,b),locx,locy,path+'/DIDSON_dataset/lamprey_label'+picture['name'][42:],item['box2d']['x1'],item['box2d']['y1'],w,h,distance])


    df=pd.DataFrame(lamprey_list,columns=['ID','x1','x2','y1','y2','minlength','Maxlength','location_x','location_y','file','x','y','w','h','distance'])
    df4=df.sort_values(by=['file'])
    indexNames = df4[ df4['file'] == path+'/DIDSON_dataset/lamprey_label/lamprey/2013-05-21_003000_HF_S001_14.png' ].index
    df4.drop(indexNames , inplace=True)
    indexNames = df4[ df4['file'] == path+'/DIDSON_dataset/lamprey_label/lamprey/2013-05-21_003000_HF_S001_13.png' ].index
    df4.drop(indexNames , inplace=True)
    return df4

# sucker data part
def read_sucker(sucker_data,path):
    sucker_list =[]
    for picture in sucker_data['frames']:
        k=len(picture['labels'])
        for item in picture['labels']:
            if(k==1):
                a=(item['box2d']['x1'],item['box2d']['y1'])#pixel
                b=(item['box2d']['x2'],item['box2d']['y2'])
                short=item['box2d']['x2']-item['box2d']['x1']
                w=item['box2d']['x2']-item['box2d']['x1']
                h=item['box2d']['y2']-item['box2d']['y1']
                long=dist(a,b)
                locx=statistics.mean([item['box2d']['x1'],item['box2d']['x2']]) #bounding box中心點X
                locy=statistics.mean([item['box2d']['y1'],item['box2d']['y2']])#bounding box中心點y, 坐標軸原點在左上角
                distance=dist((locx,locy),(200,0))
                sucker_list.append([item['id'],item['box2d']['x1'],item['box2d']['x2'],item['box2d']['y1'],item['box2d']['y2'],item['box2d']['x2']-item['box2d']['x1'],
                            dist(a,b),locx,locy,path+'/DIDSON_dataset/sucker_label'+picture['name'][41:],item['box2d']['x1'],item['box2d']['y1'],w,h,distance])


    df=pd.DataFrame(sucker_list,columns=['ID','x1','x2','y1','y2','minlength','Maxlength','location_x','location_y','file','x','y','w','h','distance'])
    df5=df.sort_values(by=['file'])
    return df5

# read pike and lmbass
def read_pike_lmbass(path, subfolder, image_path):
    pike_list=[]
    lmbass_list=[]
    list=[]
    
    for k in subfolder:
        folder=os.listdir(path+'/'+k)
        for file in folder:
            xml= ET.parse(path+'/'+k+'/'+file)
            root=xml.getroot()
            filename=root.find('filename').text
            obj=root.findall('object')
            for i in obj:
                name=i.find('name').text
                bndbox=i.find('bndbox')
                xmin=float(bndbox[0].text)
                ymin=float(bndbox[1].text)
                xmax=float(bndbox[2].text)
                ymax=float(bndbox[3].text)
                x=xmin
                y=ymin
                w=xmax-xmin
                h=ymax-ymin
                locx=statistics.mean([xmin,xmax])
                locy=672-statistics.mean([ymin,ymax])

                long=dist((xmin,ymin),(xmax,ymax))
                short=xmax-xmin
                if name=='pike':
                    if len(obj)==1:
                        TS=(28.6*(math.log10((long)*1500/(637.16*2))))-73.7
                        pike_list.append([name,(long)*1500/(637.16*2),TS,locx,locy,len(obj),image_path+k+'/'+filename,x,y,w,h])
                elif name=='lmbass':
                    if len(obj)==1:
                        TS=(15.37*(math.log10((long)*1500/(637.16*2))))-56.26
                        lmbass_list.append([name,(long)*1500/(637.16*2),TS,locx,locy,len(obj),image_path+k+'/'+filename,x,y,w,h])
                        

    df6=pd.DataFrame(pike_list,columns=['filename','average(cm)','TS','locx','locy','num','file','x','y','w','h'])
    df7=pd.DataFrame(lmbass_list,columns=['filename','average(cm)','TS','locx','locy','num','file','x','y','w','h'])
    return df6,df7

