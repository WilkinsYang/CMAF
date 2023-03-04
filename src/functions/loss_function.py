import numpy as np
import tensorflow as tf
from keras import backend as K
from layers.masking_strategy import *
from .split_data import class_weight

def FocalLoss(gamma,alpha, y_true, y_pred):
        epsilon = 1.e-9
        alpha = np.array(alpha, dtype='float32')
        y_true = tf.convert_to_tensor(y_true, tf.float32)
        y_pred = tf.convert_to_tensor(y_pred, tf.float32)

        model_out = tf.add(y_pred, epsilon)
        ce = tf.multiply(y_true, -tf.math.log(model_out))
      
        weight = tf.multiply(y_true, (tf.subtract(1., model_out)))
        weight = K.clip(weight, K.epsilon(), 1.0)
        weight=tf.pow(weight,gamma)
                 
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return tf.reduce_mean(reduced_fl)


#custom loss function for feature mining
def custom_loss(y_true,y_pred):
    global gamma1
    a1,a2,a3=tf.split(y_pred, num_or_size_splits=3, axis=1)
    
    cce1=FocalLoss(0.5, class_weight,y_true, a1)
    cce2=FocalLoss(0.5+gamma1[0,0], class_weight,y_true, a2)
    cce3=FocalLoss(0.5+gamma1[0,1], class_weight,y_true, a3)
    
    # summation of three paths
    cce=cce1+cce2+cce3
    return cce
