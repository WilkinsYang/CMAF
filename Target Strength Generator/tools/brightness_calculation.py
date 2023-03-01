import math
import cv2
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np


def brightness(df):
    list=[]
    for index,row in df.iterrows():
        image=cv2.imread(row['file'],cv2.IMREAD_GRAYSCALE)
        x = math.floor(row['x'])
        y = math.floor(row['y'])
        w = math.ceil(row['w'])
        h = math.ceil(row['h'])
        sum=0
        crop_img=image[y:y+h, x:x+w]
        hist = cv2.calcHist([crop_img], [0], None, [256], [0, 256])
        crop_img=crop_img.reshape(h*w,1)
        kmeans = KMeans(n_clusters=4, random_state=0).fit(crop_img)
        #print('Kmeans:',kmeans.cluster_centers_)
        score=kmeans.cluster_centers_
        score=np.sort(score,axis=0)
        #print((score[1]+score[2])/2)
        threshold=(score[2]+score[3])/2
        
        area=0
        for i in crop_img:
            if(i>=threshold):
                #print(type(i))
                sum=sum+i.astype(int)
                area=area+1
        #sum=np.expand_dims(sum,axis=0)
        #print(sum.size)
        list.append([row['file'],row['location_x'],row['location_y'],sum[0]/area,area,row['distance'],row['TS']])


    df_total=pd.DataFrame(list, columns=['filename','locx','locy','brightness','area','distance','TS'])
    return df_total

def brightness_no_TS(df):
    list=[]
    for index,row in df.iterrows():
        image=cv2.imread(row['file'],cv2.IMREAD_GRAYSCALE)
        x = math.floor(row['x'])
        y = math.floor(row['y'])
        w = math.ceil(row['w'])
        h = math.ceil(row['h'])
        sum=0
        crop_img=image[y:y+h, x:x+w]
        hist = cv2.calcHist([crop_img], [0], None, [256], [0, 256])
        crop_img=crop_img.reshape(h*w,1)
        kmeans = KMeans(n_clusters=4, random_state=0).fit(crop_img)
        #print('Kmeans:',kmeans.cluster_centers_)
        score=kmeans.cluster_centers_
        score=np.sort(score,axis=0)
        #print((score[1]+score[2])/2)
        threshold=(score[2]+score[3])/2
        
        area=0
        for i in crop_img:
            if(i>=threshold):
                #print(type(i))
                sum=sum+i.astype(int)
                area=area+1
        #sum=np.expand_dims(sum,axis=0)
        #print(sum.size)
        list.append([row['file'],row['location_x'],row['location_y'],sum[0]/area,area,row['distance']])


    df_total=pd.DataFrame(list, columns=['filename','locx','locy','brightness','area','distance'])
    return df_total
