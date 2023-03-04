from tensorflow import keras
from keras.applications.vgg16 import VGG16
import pandas as pd 
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from matplotlib import colors

from functions import *
from layers import *


print(tf.executing_eagerly())
#tf.compat.v1.Session()
#%% parameters
np.random.seed(10)
FISH_NUMBER=6
after_reshape=[]
PATH='your_path'    #modify your path here
EPOCH=100
#%% read data
df_smbass = pd.read_csv(PATH+'/smbass_output_1fish.csv')
df_carp = pd.read_csv(PATH+'/carp_output_1fish.csv')
df_walleye = pd.read_csv(PATH+'/walleye_output_1fish.csv')
df_lamprey= pd.read_csv(PATH+'/lamprey_output_1fish.csv')
df_sucker= pd.read_csv(PATH+'/sucker_output_1fish.csv')
df_pike=pd.read_csv(PATH+'/pike_output_1fish.csv')
df_lmbass=pd.read_csv(PATH+'/lmbass_output_1fish.csv')
df_smbass.columns = range(df_smbass.shape[1])
df_carp.columns = range(df_carp.shape[1])
df_walleye.columns = range(df_walleye.shape[1])
df_lamprey.columns = range(df_lamprey.shape[1])
df_sucker.columns = range(df_sucker.shape[1])
df_pike.columns = range(df_pike.shape[1])
df_lmbass.columns = range(df_lmbass.shape[1])

## split training, validation, and testing data
test_path_list, train_path_list, validation_path_list, train_signal_feature,test_signal_feature, validation_signal_feature, test_label_onehot, train_label_onehot, validation_label_onehot, FISH_NUMBER = split(df_smbass, df_carp, 
                                                                                                                                                                                                                df_walleye, df_lamprey, df_sucker, df_pike, df_lmbass, FISH_NUMBER)
print('finish split data')

#%% dataloader
params={'batch_size':1}
training_generator = dataloader(train_path_list, train_signal_feature, train_label_onehot,**params)
test_generator = dataloader(test_path_list, test_signal_feature, test_label_onehot,1)
validation_generator = dataloader(validation_path_list,validation_signal_feature,validation_label_onehot,**params)

## initial masking ratio
gamma1=np.array([[0.5, 0.5]])

#%% model
input1 = tf.keras.layers.Input(shape=(40,40,3))
input2 = tf.keras.layers.Input(shape=(300,))
# image featrue ectractor
cnn_out=VGG16(include_top=False, input_shape=(40,40,3),pooling='avg')(input1)
i=tf.keras.layers.Flatten()(cnn_out)
# cross modal fusion
fusion=fusion_layer()
att_visual_features,att_impulse_features=fusion([i, input2])
merged=tf.keras.layers.Concatenate(axis=-1,trainable=False)([att_visual_features,att_impulse_features])

## training strategy
# masking layer
mask_result, mask_result2=masking_layer()(merged)
# three classifier
feature_mining=feature_mining_layer()
out1, out2, out3 = feature_mining([merged, mask_result, mask_result2])
#output layer
outputs=tf.keras.layers.Concatenate(axis=-1,trainable=False)([out1,out2,out3])

model = tf.keras.models.Model(inputs=[input1, input2], outputs=outputs)
opt = tf.keras.optimizers.Adam(learning_rate=1e-06)
# run eagerly must turn on
model.compile(loss=MBFL, optimizer = opt, metrics=['accuracy'],run_eagerly=True) 
model.summary()
Earlystop=EarlyStopping(monitor='val_loss',patience=8,verbose=1,restore_best_weights=True)
tqdm_callback = run_time(EPOCH)

# start training
train_history = model.fit(x=training_generator, epochs=EPOCH ,callbacks=[Earlystop,tqdm_callback],validation_data=validation_generator,verbose=1)

#%% training results
show_train_history(train_history, 'accuracy', 'val_accuracy')
show_train_history(train_history, 'loss', 'val_loss')

# evaluation results
scores = model.evaluate(test_generator)
print('test loss, test accuracy=', scores)

y_pred1 = np.zeros(len(test_generator))
y_pred2 = np.zeros(len(test_generator))
y_pred3 = np.zeros(len(test_generator))
y_true = np.zeros(len(test_generator))

output = model.predict_generator(test_generator)
output1=output[:,0:6]
output2=output[:,6:12]
output3=output[:,12:18]

for i in range(len(output)):
    y_pred1[i] = np.argmax(output1[i])
    y_pred2[i] = np.argmax(output2[i])
    y_pred3[i] = np.argmax(output3[i])
    
    y_true[i] = np.argmax(test_generator[i][1])
cm1 = confusion_matrix(y_true, y_pred1) #path1
cm2 = confusion_matrix(y_true, y_pred2) #path2
cm3 = confusion_matrix(y_true, y_pred3) #path3

fig, ax = plt.subplots()
plot_confusion_matrix(cm1, 'Confusion matrix (path1)')
plot_confusion_matrix(cm2, 'Confusion matrix (path2)')
plot_confusion_matrix(cm3, 'Confusion matrix (path3)')

cmn = 100*cm1.astype('float') / cm1.sum(axis=1)[:, np.newaxis]
confusion_matrix2(cmn)

#%% show F1-score
target_names = ['Bass', 'Carp', 'Walleye','Sucker', 'Lamprey', 'Pike']
print(classification_report(y_true, y_pred1, target_names=target_names,digits=4))
print('balanced accuracy:',balanced_accuracy_score(y_true, y_pred1))


