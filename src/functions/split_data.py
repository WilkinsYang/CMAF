import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.utils import np_utils
import numpy as np

class_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

def split(df_smbass, df_carp, df_walleye, df_lamprey, df_sucker, df_pike, df_lmbass, FISH_NUMBER):
    pike_train=df_pike.iloc[:191,:]
    pike_test=df_pike.iloc[191:,:]
    df_sucker1, df_sucker2=train_test_split(df_sucker,random_state=7,train_size=0.3)
    # you can decide to choose different class of fish, 6 denotes with imbalance problem, 4 denotes without
    if(FISH_NUMBER==6):
        df=pd.concat([df_smbass,df_carp,df_walleye,df_sucker1, df_lamprey,df_lmbass],axis=0,ignore_index=True)
    elif (FISH_NUMBER==4):
        df=pd.concat([df_smbass,df_carp,df_walleye,df_lmbass],axis=0,ignore_index=True)
    df=df.sample(frac=1).reset_index(drop=True) #把data打散
    df.iloc[:,6:306]=preprocessing.scale(df.iloc[:,6:306]) #normalize
    
    # train test split
    train_data, test_data = train_test_split(df, random_state=4, train_size = 0.8)
    train_data=pd.concat([train_data,pike_train],axis=0,ignore_index=True)
    test_data=pd.concat([test_data,pike_test],axis=0,ignore_index=True)
    # 4 fish need to change the label in dataset
    if(FISH_NUMBER==4):
        for i in range(train_data.shape[0]):
            if (train_data.iloc[i,5]==5):
                train_data.iloc[i,5]=3
        for j in range(test_data.shape[0]):
            if(test_data.iloc[j,5]==5):
                test_data.iloc[j,5]=3

    ## testing data
    # visual data
    test_path_list=test_data.iloc[:,0]
    test_path_list=test_path_list.to_numpy()
    # sonar signal data
    test_signal_feature=test_data.iloc[:,1:306]
    test_signal_feature=test_signal_feature.to_numpy()
    test_data_label=test_data.iloc[:,5]
    # one hot encoding
    test_label_onehot = np_utils.to_categorical(test_data_label)

    train_data, validation_data = train_test_split(train_data, random_state=46, train_size = 0.7)
    ## training data
    # visual data
    train_path_list=train_data.iloc[:,0]
    train_path_list=train_path_list.to_numpy()
    # sonar signal data
    train_signal_feature=train_data.iloc[:,1:306]
    train_signal_feature=train_signal_feature.to_numpy()
    train_data_label=train_data.iloc[:,5]
    # one hot encoding
    train_label_onehot = np_utils.to_categorical(train_data_label)

    #validation data
    # visual data
    validation_path_list=validation_data.iloc[:,0]
    validation_path_list=validation_path_list.to_numpy()
    # sonar signal data
    validation_signal_feature=validation_data.iloc[:,1:306]
    validation_signal_feature=validation_signal_feature.to_numpy()
    validation_data_label=validation_data.iloc[:,5]
    # one hot encoding
    validation_label_onehot = np_utils.to_categorical(validation_data_label)

    # conventional class weight (inverse of the sample number of each class)
    weight(train_data_label, FISH_NUMBER)
    return test_path_list, train_path_list, validation_path_list, train_signal_feature,test_signal_feature, validation_signal_feature, test_label_onehot, train_label_onehot, validation_label_onehot, FISH_NUMBER


def weight(train_data_label, FISH_NUMBER):
    global class_weight
    if(FISH_NUMBER==6):
        class_weight=[(train_data_label==0).sum(),(train_data_label==1).sum(),(train_data_label==2).sum(),(train_data_label==3).sum(),
                    (train_data_label==4).sum(),(train_data_label==5).sum()]
    elif(FISH_NUMBER==4):
        class_weight=[(train_data_label==0).sum(),(train_data_label==1).sum(),(train_data_label==2).sum(),(train_data_label==3).sum()]

    class_weight=np.array(class_weight)
    class_weight=1/class_weight

    