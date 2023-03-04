from keras import backend as K
from tensorflow import keras
import tensorflow as tf
import numpy as np

# initial value
gamma1=np.array([[0.5, 0.5]])

class CustomLayer(keras.layers.Layer):
    #@tf.function # not available when turn "run eagerly" to true
    def call(self,inputs,training=None):
        if training is None:
            training = K.learning_phase()
        if training:
            global gamma1
            b=tf.ones((1,812),tf.int32)
            #隨機產生mask ratio
            prob=tf.random.uniform((1,1), minval=0,maxval=1,dtype=tf.float32, seed=None, name=None)
            prob_total=tf.concat([prob, 1.-prob],1)          

            #根據mask ratio產生mask
            a=tf.random.categorical(tf.math.log(prob_total),812, dtype=tf.int32)
            b=tf.math.subtract(b,a)
                        
            ##傳mask ratio出去
            gamma1=prob_total.numpy()
                      
            a= tf.dtypes.cast(a, tf.float32)
            b= tf.dtypes.cast(b, tf.float32)
            mask1=tf.math.multiply(a,inputs)
            mask2=tf.math.multiply(b,inputs)
            
            return mask1, mask2
        else:
            return inputs,inputs
            