## import package
import os
import json
from tensorflow import keras
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from tools import *
import predictor

## Modify your path here
PATH='your_path'

## read data
#walleye, smbass, carp, lamprey, sucker
file1 = open(PATH+'/DIDSON_dataset/walleye_label_5028/DIDSON_walleye_nl_export_2021-11-07_09-36-02.json')
walleye_data = json.load(file1)
file2 = open(PATH+'/DIDSON_dataset/smbass_label_5040/DIDSON_smbass_nl_export_2021-11-07_09-35-46.json')
smbass_data = json.load(file2)
file3 = open(PATH+'/DIDSON_dataset/carp_label_5095/DIDSON_carp_nl_export_2021-11-07_09-33-58.json')
carp_data = json.load(file3)
file4 = open(PATH+'/DIDSON_dataset/lamprey_label/DIDSON_lamprey_export_2021-10-15_16-31-59.json')
lamprey_data = json.load(file4)
file5 = open(PATH+'/DIDSON_dataset/sucker_label/DIDSON_sucker_export_2021-11-07_09-36-32.json')
sucker_data = json.load(file5)

#Pike, lmbass
path = PATH+'/didson_custom_extra/anno'
subfolder=os.listdir(path)
image_path=PATH+'/didson_custom_extra/Data/'


## data preparation
df1=read_walleye(walleye_data,PATH)
df2=read_smbass(smbass_data,PATH)
df3=read_carp(carp_data,PATH)
df4=read_lamprey(lamprey_data,PATH)
df5=read_sucker(sucker_data,PATH)
df6, df7=read_others(path, subfolder, image_path)
print('finish data preparation')
print('****************************************')

## brightness and area idenification
df1_total=brightness(df1)
df2_total=brightness(df2)
df3_total=brightness(df3)
df4_total=brightness_no_TS(df4)
df5_total=brightness_no_TS(df5)
print('finish brightness and area identification')
print('****************************************')

## Prediction model
data = pd.concat([df1_total,df2_total,df3_total],axis=0,ignore_index=True)
data.pop('filename')
data.pop('distance')
#split training and testing data
train_data,test_data=train_test_split(data, random_state=77, train_size = 0.8)
train_labels=train_data.pop('TS')
test_labels=test_data.pop('TS')


normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_data))


#training
dnn_model = predictor.predictor(normalizer)
dnn_model.summary()

history = dnn_model.fit(
    train_data,
    train_labels,
    validation_split=0.2,
    verbose=1, epochs=200)

plot_loss(history)

#evaluation
test_results = {}
test_results['dnn_model'] = dnn_model.evaluate(test_data, test_labels, verbose=0)
test_predictions = dnn_model.predict(test_data).flatten()
error = test_predictions - test_labels
#plot evaluation results
plot_distribution(test_labels, test_predictions)

plot_histogram(error)
print('finish evaluation')
print('****************************************')


## generate TS of data without ground truth
sucker_test=df5_total
sucker_test.pop('filename')
sucker_test.pop('distance')
lamprey_test=df4_total
lamprey_test.pop('filename')
lamprey_test.pop('distance')
sucker_predictions = dnn_model.predict(sucker_test).flatten()
df=pd.DataFrame(sucker_predictions,columns=['TS'])
df_sucker=pd.merge(df5,df,left_index=True, right_index=True)
lamprey_predictions = dnn_model.predict(lamprey_test).flatten()
df=pd.DataFrame(lamprey_predictions,columns=['TS'])
df_lamprey=pd.merge(df4,df,left_index=True, right_index=True)
print('finish prediction')
print('****************************************')


## output data for next stage (Sonar emulator)
df_walleye=out_data(df1)
df_smbass=out_data(df2)
df_carp=out_data(df3)
df_lamprey=out_data(df_lamprey)
df_sucker=out_data(df_sucker)

df_walleye.to_csv(PATH+'\DIDSON_dataset\walleye_result_1fish.csv')
df_smbass.to_csv(PATH+'\DIDSON_dataset\smbass_result_1fish.csv')
df_carp.to_csv(PATH+'\DIDSON_dataset\carp_result_1fish.csv')
df_lamprey.to_csv(PATH+'\DIDSON_dataset\lamprey_result_1fish.csv')
df_sucker.to_csv(PATH+'\DIDSON_dataset\sucker_result_1fish.csv')
df6.to_csv(PATH+'\didson_custom_extra\pike_result_1fish.csv',index=False)
df7.to_csv(PATH+'\didson_custom_extra\lmbass_result_1fish.csv',index=False)
print('finish output data')
print('****************************************')
