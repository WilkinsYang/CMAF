from tensorflow import keras
import math
import cv2
import numpy as np

#dataloader (with batch)
class Image_path_Dataloader_custom(keras.utils.Sequence):
    def __init__(self, img_path_list,img_signal, img_label,batch_size):
        self.img_path = img_path_list
        self.img_feature=img_signal
        self.label=img_label
        self.batch_size=batch_size
    # override the __getitem__ method. this is the method that dataloader calls
    def __len__(self):
        return int(np.ceil(len(self.img_path))/float(self.batch_size))
    def __getitem__(self, index):
        batch_impulse=self.img_feature[index*self.batch_size:(index+1)*self.batch_size]
        batch_path=self.img_path[index*self.batch_size:(index+1)*self.batch_size]
        batch_y=self.label[index*self.batch_size:(index+1)*self.batch_size]
        impulse=batch_impulse[:,5:305]
        #for file_name in batch_path:
        x1=[]
        for idx, file_name in enumerate(batch_path):
            image = cv2.imread(file_name)
            x = math.floor(batch_impulse[idx][0])
            y = math.floor(batch_impulse[idx][1])
            w = math.ceil(batch_impulse[idx][2])
            h = math.ceil(batch_impulse[idx][3])
            crop_img=image[y:y+h, x:x+w]
            crop_img = cv2.resize(crop_img, (40, 40), interpolation=cv2.INTER_CUBIC)
            crop_img=crop_img/255

            x1.append(crop_img)
            
        
        return [np.array(x1),np.array(impulse)],batch_y